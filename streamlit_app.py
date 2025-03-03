import streamlit as st
import openai
import os
from dotenv import load_dotenv

# Carica la chiave API da .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Funzione per chiamare OpenAI con streaming
def chat_with_gpt(prompt, chat_history):
    messages = chat_history + [{"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        stream=True  # Attiva lo streaming della risposta
    )

    full_response = ""
    for chunk in response:
        if "choices" in chunk and len(chunk["choices"]) > 0:
            full_response += chunk["choices"][0]["delta"].get("content", "")
            yield full_response  # Restituisce la risposta in streaming

# Interfaccia Streamlit
st.set_page_config(page_title="Chatbot con OpenAI", page_icon="💬")

st.title("💬 Chatbot con GPT")
st.write("Interagisci con OpenAI direttamente da questa interfaccia!")

# Inizializza la sessione per la chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "Sei un assistente AI amichevole."}]

# Mostra la chat esistente
for message in st.session_state.chat_history:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Area di input utente
user_input = st.text_input("Scrivi un messaggio:", "")

if st.button("Invia") and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Streaming della risposta
    with st.chat_message("assistant"):
        response_stream = chat_with_gpt(user_input, st.session_state.chat_history)
        full_response = st.write_stream(response_stream)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
