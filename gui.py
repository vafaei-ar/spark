import gradio as gr
import requests

API_URL = "http://localhost:8000/chat"

# Session state
chat_history = []

def chat_with_api(message, chat_history, patient_id, model_name):
    # Send message to API
    try:
        response = requests.post(API_URL, json={
            "model": model_name,
            "patient_id": patient_id,
            "question": message
        })

        if response.status_code == 200:
            answer = response.json()["answer"]
        else:
            answer = f"[Error {response.status_code}] {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        answer = f"[Exception] {str(e)}"

    # Return updated history
    chat_history.append((message, answer))
    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("## 🧑‍⚕️ Patient Q&A Chatbot (RAG via API)")
    
    with gr.Row():
        patient_id = gr.Textbox(label="Patient ID", placeholder="e.g. PSU30395013", value="PSU30395013")
        model_name = gr.Textbox(label="Model", placeholder="e.g. gemma3", value="gemma3")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Question", placeholder="Ask about medications, diagnoses...", lines=1)
    submit_btn = gr.Button("Send")

    clear_btn = gr.Button("Clear Chat")

    def clear_chat():
        return [], ""

    submit_btn.click(
        fn=chat_with_api,
        inputs=[msg, chatbot, patient_id, model_name],
        outputs=[msg, chatbot]
    )

    clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbot, msg])

demo.launch()
