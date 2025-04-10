import os
import json
import uuid
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
DATA_DIR = "data/"  # Directory containing parquet files
CHROMA_DB_BASE = "./chromadb/patient_dbs"
# TABLES = [
#     "condition", "death", "death_cause", "demographic", "diagnosis",
#     "dispensing", "encounter", "immunization", "lab_result_cm", "med_admin",
#     "obs_clin", "obs_gen", "prescribing", "procedures", "vital"
# ]
TABLES = [
    "death", "demographic", "diagnosis",
    "encounter", "lab_result_cm", "procedures", "vital"
]
# EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
OLLAMA_API_URL = "http://localhost:11434/api/generate"
TOP_K_RESULTS = 5
MAX_BATCH_SIZE = 5000  # Under ChromaDB's batch limit

# --- FastAPI App ---
app = FastAPI(title="Patient Data Chat API (RAG)")

def get_ollama_embedding(text: str, model: str = "nomic-embed-text:latest", cache: dict = None) -> list:
    if cache is not None and text in cache:
        return cache[text]

    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text}
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")

        if cache is not None:
            cache[text] = embedding

        return embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None

EMBED_CACHE_DIR = "./embedding_cache"
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

def load_embedding_cache(patid):
    cache_path = os.path.join(EMBED_CACHE_DIR, f"patient_{patid}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}

def save_embedding_cache(patid, cache_dict):
    cache_path = os.path.join(EMBED_CACHE_DIR, f"patient_{patid}.json")
    with open(cache_path, "w") as f:
        json.dump(cache_dict, f)

# # --- Load Embedding Model ---
# print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
# embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
# print("Embedding model loaded.")

# --- Custom JSON Encoder ---
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)

# --- API Request Model ---
class ChatRequest(BaseModel):
    model: str
    patient_id: str
    question: str

# --- Helper: Format Record for Embedding ---
def format_record_for_embedding(record_dict, table_name):
    filtered_record = {k: v for k, v in record_dict.items() if k != 'PATID'}
    try:
        record_str = json.dumps(filtered_record, cls=CustomJSONEncoder)
        return f"Table: {table_name}, Record: {record_str}"
    except Exception as e:
        simple_repr = ", ".join([f"{k}: {v}" for k, v in filtered_record.items()])
        return f"Table: {table_name}, Record: {simple_repr}"

# --- API Endpoint ---
@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"\nReceived request for PATID: {request.patient_id}")

    # Setup per-patient ChromaDB path
    patient_db_path = os.path.join(CHROMA_DB_BASE, f"patient_{request.patient_id}")
    os.makedirs(patient_db_path, exist_ok=True)

    client = chromadb.PersistentClient(path=patient_db_path)
    collection_name = "patient_data_rag"

    # Try loading collection
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Loaded existing ChromaDB for patient {request.patient_id}")
    except:
        print(f"Creating new ChromaDB for patient {request.patient_id}")
        collection = client.create_collection(name=collection_name)

        docs, metadatas, ids = [], [], []

        for table in TABLES:
            path = os.path.join(DATA_DIR, f"{table}.parquet")
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path)
                    if "PATID" not in df.columns:
                        continue
                    df["PATID"] = df["PATID"].astype(str)
                    df_patient = df[df["PATID"] == request.patient_id]

                    for idx, record in df_patient.iterrows():
                        record_dict = record.to_dict()
                        doc_text = format_record_for_embedding(record_dict, table)
                        metadata = {"PATID": request.patient_id, "table": table}
                        doc_id = f"{table}_{request.patient_id}_{idx}_{uuid.uuid4()}"
                        docs.append(doc_text)
                        metadatas.append(metadata)
                        ids.append(doc_id)

                except Exception as e:
                    print(f"Error processing {path}: {e}")

        if not docs:
            raise HTTPException(status_code=404, detail=f"No data found for Patient ID {request.patient_id}.")

        # Generate and batch-add embeddings
        print(f"Generating embeddings for {len(docs)} records...")
        # embeddings = embedding_model.encode(docs, show_progress_bar=True).tolist()
        embedding_cache = load_embedding_cache(request.patient_id)
        embeddings = []

        for i, doc in enumerate(docs):
            emb = get_ollama_embedding(doc, cache=embedding_cache)
            if emb is not None:
                embeddings.append(emb)
            else:
                raise HTTPException(status_code=500, detail=f"Failed to embed doc index {i}")

        # Save updated cache
        save_embedding_cache(request.patient_id, embedding_cache)

        for i in range(0, len(docs), MAX_BATCH_SIZE):
            collection.add(
                embeddings=embeddings[i:i+MAX_BATCH_SIZE],
                documents=docs[i:i+MAX_BATCH_SIZE],
                metadatas=metadatas[i:i+MAX_BATCH_SIZE],
                ids=ids[i:i+MAX_BATCH_SIZE]
            )
        print(f"ChromaDB created for patient {request.patient_id}.")

    # --- Embed query ---
    try:
        print("Embedding question...")
        # query_embedding = embedding_model.encode([request.question])[0].tolist()
        query_embedding = get_ollama_embedding(request.question)
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to embed the query.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed question: {e}")

    # --- Query ChromaDB ---
    try:
        print(f"Querying ChromaDB for patient {request.patient_id}...")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=20*TOP_K_RESULTS,
            where={"PATID": request.patient_id}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve from ChromaDB: {e}")

    retrieved_docs = results['documents'][0] if results and results['documents'] else []

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail=f"No relevant data found for Patient ID {request.patient_id}.")

    # --- Build context and prompt ---
    context_str = "Relevant Patient Data Snippets:\n"
    for i, doc in enumerate(retrieved_docs):
        context_str += f"Snippet {i+1}:\n{doc}\n\n"

    prompt = (
        f"Based *only* on the following relevant data snippets provided for Patient ID {request.patient_id}, "
        f"answer the user's question.\n\n"
        f"--- Relevant Data ---\n{context_str}--- End Relevant Data ---\n\n"
        f"Question: {request.question}\n\nAnswer:"
    )

    print("\n--- Prompt for LLM ---")
    print(prompt)
    print("--- End Prompt ---\n")

    # --- Call Ollama ---
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": request.model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "No response field returned.")
        print("Ollama call successful.")
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Ollama API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    return {"answer": answer}

# --- Run App Locally ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
