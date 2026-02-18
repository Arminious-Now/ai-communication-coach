import streamlit as st
import google.generativeai as genai
from deepgram import DeepgramClient, SpeakOptions, PrerecordedOptions
from pinecone import Pinecone
from audio_recorder_streamlit import audio_recorder
import os
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Comm Coach", page_icon="🎙️")

st.title("🎙️ AI Communication Coach")
st.write("Switch between modes to practice or get advice.")

# --- SIDEBAR: SETTINGS & INGESTION ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # We use Streamlit Secrets for keys in production, but inputs for testing if needed
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

    st.divider()
    
    st.subheader("📚 Add Knowledge")
    yt_url = st.text_input("YouTube URL to Ingest")
    if st.button("Ingest Video"):
        st.info("Ingestion logic would go here (connecting to Pinecone).")
        # We will add the ingestion logic in the next step once the app is live.

# --- MAIN APP LOGIC ---

if gemini_key and deepgram_key and pinecone_key:
    
    # Initialize Clients
    genai.configure(api_key=gemini_key)
    dg_client = DeepgramClient(deepgram_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index("coach-memory")

    # Mode Selection
    mode = st.radio("Select Mode:", ["🧠 Coach Mode (Advice)", "🎭 Practice Mode (Roleplay)"], horizontal=True)

    # Audio Recorder (Works on Mobile!)
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
            # 1. Transcribe with Deepgram
            # We have to write bytes to a temp file for Deepgram SDK 
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                fp.write(audio_bytes)
                temp_filename = fp.name

            with open(temp_filename, "rb") as audio_file:
                buffer_data = audio_file.read()

            payload = { "buffer": buffer_data }
            options = PrerecordedOptions(model="nova-2", smart_format=True)
            response = dg_client.listen.rest.v("1").transcribe_file(payload, options)
            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
            
            st.success(f"You said: {transcript}")

            # 2. Logic (Gemini + Pinecone)
            ai_text_response = ""
            
            if "Coach" in mode:
                # Retrieve from Pinecone (Placeholder for now)
                context = "User is asking for advice." 
                prompt = f"Context: {context}. User Question: {transcript}. Provide brief coaching advice."
                model = genai.GenerativeModel('gemini-2.5-flash')
                res = model.generate_content(prompt)
                ai_text_response = res.text
                
            else:
                # Practice Mode
                prompt = f"You are a roleplay partner. User said: {transcript}. Respond in character (2 sentences max)."
                model = genai.GenerativeModel('gemini-2.5-flash')
                res = model.generate_content(prompt)
                ai_text_response = res.text

            st.write(f"**AI:** {ai_text_response}")

            # 3. Text to Speech (Deepgram)
            with st.spinner("Generating Audio..."):
                speak_options = SpeakOptions(model="aura-asteria-en")
                filename = "output_audio.wav"
                dg_client.speak.rest.v("1").save(filename, {"text": ai_text_response}, speak_options)
                
                st.audio(filename, format="audio/wav", autoplay=True)

else:
    st.warning("Please enter all API keys in the sidebar or secrets to proceed.")
