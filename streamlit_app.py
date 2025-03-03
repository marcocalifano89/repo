import streamlit as st
import openai
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

# Inizializza il client OpenAI con l'ultima sintassi
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Funzione per la chat con OpenAI (streaming attivato)
def chat_with_gpt(prompt, chat_history):
    messages = chat_history + [{"role": "user", "content": prompt}]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True  # Streaming della risposta
    )

    full_response = ""  # Buffer della risposta
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_response += text  # Accumula la risposta
            yield text  # Restituisce solo il nuovo testo

    yield "\n"  # Assicura una fine pulita della risposta

# Configura la UI di Streamlit
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

# Input dell'utente
user_input = st.text_input("Scrivi un messaggio:", "")

if st.button("Invia") and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Streaming della risposta
    full_response = ""
    with st.chat_message("assistant"):
        response_stream = chat_with_gpt(user_input, st.session_state.chat_history)
        response_container = st.empty()  # Crea un contenitore per aggiornare il testo
        for text in response_stream:
            full_response += text
            response_container.markdown(full_response)  # Aggiorna la risposta man mano

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
