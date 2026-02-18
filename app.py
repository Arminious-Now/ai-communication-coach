import streamlit as st
import google.generativeai as genai
from deepgram import DeepgramClient
from pinecone import Pinecone
from audio_recorder_streamlit import audio_recorder
import os
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Comm Coach", page_icon="🎙️")

st.title("🎙️ AI Communication Coach")
st.write("Switch between modes to practice or get advice.")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    if 'GEMINI_KEY' in st.secrets:
        gemini_key = st.secrets['GEMINI_KEY']
    else:
        gemini_key = st.text_input("Gemini API Key", type="password")

    if 'DEEPGRAM_KEY' in st.secrets:
        deepgram_key = st.secrets['DEEPGRAM_KEY']
    else:
        deepgram_key = st.text_input("Deepgram API Key", type="password")
        
    if 'PINECONE_KEY' in st.secrets:
        pinecone_key = st.secrets['PINECONE_KEY']
    else:
        pinecone_key = st.text_input("Pinecone API Key", type="password")

# --- MAIN APP LOGIC ---

if gemini_key and deepgram_key and pinecone_key:
    
    # Initialize Clients
    genai.configure(api_key=gemini_key)
    # Deepgram v5 Client
    dg_client = DeepgramClient(deepgram_key) 
    
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index("coach-memory")

    # Mode Selection
    mode = st.radio("Select Mode:", ["🧠 Coach Mode (Advice)", "🎭 Practice Mode (Roleplay)"], horizontal=True)

    # Audio Recorder
    audio_bytes = audio_recorder(
        text="Click to Record",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x",
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        with st.spinner("Transcribing..."):
            # 1. Transcribe with Deepgram (v5 Syntax)
            # We must use a file-like object or bytes for v5
            payload = {"buffer": audio_bytes}
            
            # UPDATED: v5 Syntax uses method chaining, no "PrerecordedOptions" class needed
            options = {
                "model": "nova-2", 
                "smart_format": True
            }
            
            response = dg_client.listen.rest.v("1").transcribe_file(payload, options)
            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
            
            st.success(f"You said: {transcript}")

            # 2. Logic (Gemini + Pinecone)
            ai_text_response = ""
            
            if "Coach" in mode:
                # Placeholder RAG logic
                prompt = f"User Question: {transcript}. Provide brief coaching advice."
                model = genai.GenerativeModel('gemini-2.0-flash') # Updated model name
                res = model.generate_content(prompt)
                ai_text_response = res.text
                
            else:
                # Practice Mode
                prompt = f"You are a roleplay partner. User said: {transcript}. Respond in character (2 sentences max)."
                model = genai.GenerativeModel('gemini-2.0-flash')
                res = model.generate_content(prompt)
                ai_text_response = res.text

            st.write(f"**AI:** {ai_text_response}")

            # 3. Text to Speech (Deepgram v5 Syntax)
            with st.spinner("Generating Audio..."):
                filename = "output_audio.mp3"
                
                # UPDATED: v5 Syntax for TTS
                options = {
                    "model": "aura-asteria-en",
                    "text": ai_text_response
                }
                
                # This saves the file directly
                dg_client.speak.rest.v("1").save(filename, options)
                
                st.audio(filename, format="audio/mp3", autoplay=True)

else:
    st.warning("Please enter all API keys in the sidebar or secrets to proceed.")
