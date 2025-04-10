import requests

response = requests.post(
    "http://localhost:8000/chat",
    headers={"Content-Type": "application/json"},
    json={
        "model": "gemma3",
        "patient_id": "PSU30770169", 
        "question": "What diagnoses has this patient received?"
    }
)

print(response['answer'])