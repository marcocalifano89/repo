import streamlit as st
import openai
import os
import pytesseract
import numpy as np
import cv2
from dotenv import load_dotenv
from PIL import Image
import base64
import io

# Carica le variabili d'ambiente (solo per uso locale)
load_dotenv()

# Inizializza il client OpenAI con la gestione sicura della chiave API
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Funzione per convertire un'immagine in base64 per GPT-4 Turbo
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Funzione per elaborare l'immagine e migliorare l'OCR
def preprocess_image(image):
    image_cv = np.array(image)  # Converti PIL in array OpenCV
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

# Funzione per eseguire l'OCR con Tesseract
def extract_text_from_image(image):
    processed_image = preprocess_image(image)
    return pytesseract.image_to_string(processed_image, lang="eng")

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
st.set_page_config(page_title="Chatbot con OCR", page_icon="📄")

st.title("🧠 Chatbot con GPT-4 Turbo + OCR")
st.write("Interagisci con OpenAI e carica immagini per estrarre testo!")

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

# Se è stata caricata un'immagine, mostrarla ed eseguire l'OCR
image_text = ""
image_data = None

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Immagine caricata", use_column_width=True)

    # Estrarre testo con OCR
    with st.spinner("Estrazione del testo..."):
        image_text = extract_text_from_image(image)

    # Mostrare il testo estratto
    if image_text.strip():
        st.text_area("Testo estratto:", image_text)
    else:
        st.warning("Nessun testo rilevato.")

    # Convertire l'immagine per GPT-4 Turbo
    image_data = image_to_base64(image)

# Input dell'utente
user_input = st.text_input("Scrivi un messaggio:", "")

if st.button("Invia") and (user_input or uploaded_image):
    # Se c'è un testo estratto dall'immagine, aggiungerlo al prompt
    final_prompt = user_input
    if image_text.strip():
        final_prompt += f"\n[TESTO ESTRATTO DALL'IMMAGINE]: {image_text}"

    # Aggiungi la richiesta alla chat history
    st.session_state.chat_history.append({"role": "user", "content": final_prompt})

    # Streaming della risposta
    full_response = ""
    with st.chat_message("assistant"):
        response_stream = chat_with_gpt(final_prompt, st.session_state.chat_history, image_data)
        response_container = st.empty()  # Contenitore per l'aggiornamento dinamico della risposta
        for text in response_stream:
            full_response += text
            response_container.markdown(full_response)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
