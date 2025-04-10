import os
import json
import subprocess
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Patient Data Chat API")

# Custom JSON encoder to handle timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)

# List of your PCORI CDM tables (without the .parquet extension)
TABLES = [
    "condition", "death", "death_cause", "demographic", "diagnosis",
    "dispensing", "encounter", "immunization", "lab_result_cm", "med_admin",
    "obs_clin", "obs_gen", "prescribing", "procedures", "vital"
]

# Dictionary to hold DataFrames loaded from Parquet files
dataframes = {}

# Load each table from the data/ directory
for table in TABLES:
    path = os.path.join("/home/asadr/datasets/stroke_data/", f"{table}.parquet")
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            dataframes[table] = df
            print(f"Loaded {table}.parquet with {len(df)} rows.")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    else:
        print(f"Warning: {path} does not exist.")

def get_patient_data(patient_id: str):
    """
    Retrieve patient-specific data from each table based on the PATID field.
    Returns a dictionary with table names as keys and a list of record dicts as values.
    """
    patient_data = {}
    for table, df in dataframes.items():
        if "PATID" in df.columns:
            # Convert PATID to string for safe comparison
            filtered = df[df["PATID"].astype(str) == patient_id]
            if not filtered.empty:
                # Convert rows to a list of dictionaries
                patient_data[table] = filtered.to_dict(orient="records")
    return patient_data

# Pydantic model for the request body
class ChatRequest(BaseModel):
    model: str      # For example: "gemma3"
    patient_id: str
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    
    # Retrieve data for the given patient
    patient_data = get_patient_data(request.patient_id)
    if not patient_data:
        raise HTTPException(status_code=404, detail="No data found for the given patient ID.")
    
    # Build a text summary from the patient data
    context_str = "Patient Data Summary:\n"
    for table, records in patient_data.items():
        context_str += f"\nTable: {table}\n"
        for record in records:
            # Convert each record to JSON using the custom encoder
            context_str += json.dumps(record, cls=CustomJSONEncoder) + "\n"
    
    # Construct a prompt for the LLM
    prompt = (
        f"Patient ID: {request.patient_id}\n"
        f"{context_str}\n"
        f"Question: {request.question}\n"
        "Answer:"
    )
    print(prompt)
    # Call the Ollama API
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": request.model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        result = response.json()
        answer = result["response"]
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Ollama API: {e}")
    
    return {"answer": answer}

if __name__ == "__main__":
    # For local testing: run the app with uvicorn
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
