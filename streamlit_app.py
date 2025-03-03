import streamlit as st
import openai
import os
from dotenv import load_dotenv
from PIL import Image
import base64
import io

# Carica le variabili d'ambiente (solo per uso locale)
load_dotenv()

# Inizializza il client OpenAI con la gestione sicura della chiave API
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Funzione per convertire un'immagine in base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Funzione per la chat con OpenAI (supporto a testo + immagini)
def chat_with_gpt(prompt, chat_history, image_data=None):
    messages = chat_history + [{"role": "user", "content": prompt}]
    
    # Se c'è un'immagine, aggiungila alla richiesta
    if image_data:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_data}  # Invia l'immagine in base64
            ]
        })
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",  # Modello multimodale
        messages=messages,
        stream=True
    )

    full_response = ""  # Buffer per la risposta
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_response += text
            yield text  # Ritorna il testo progressivamente

    yield "\n"  # Segnale di fine risposta

# Configura l'UI di Streamlit
st.set_page_config(page_title="Chatbot con OpenAI", page_icon="💬")

st.title("💬 Chatbot con GPT-4 Turbo (Testo + Immagini)")
st.write("Interagisci con OpenAI e carica immagini per ottenere risposte avanzate!")

# Inizializza la sessione per la chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "Sei un assistente AI amichevole."}]

# Mostra la chat esistente
for message in st.session_state.chat_history:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Caricamento dell'immagine (opzionale)
uploaded_image = st.file_uploader("Carica un'immagine (opzionale):", type=["png", "jpg", "jpeg"])

# Input dell'utente
user_input = st.text_input("Scrivi un messaggio:", "")

if st.button("Invia") and (user_input or uploaded_image):
    image_data = None
    
    # Se è stata caricata un'immagine, convertirla in base64
    if uploaded_image:
        image = Image.open(uploaded_image)
        image_data = image_to_base64(image)
        st.image(image, caption="Immagine caricata", use_column_width=True)

    # Aggiungi la richiesta alla chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Streaming della risposta
    full_response = ""
    with st.chat_message("assistant"):
        response_stream = chat_with_gpt(user_input, st.session_state.chat_history, image_data)
        response_container = st.empty()  # Contenitore per l'aggiornamento dinamico della risposta
        for text in response_stream:
            full_response += text
            response_container.markdown(full_response)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
