import streamlit as st
import google.generativeai as genai
from deepgram import DeepgramClient
from pinecone import Pinecone
from audio_recorder_streamlit import audio_recorder
from youtube_transcript_api import YouTubeTranscriptApi
from pypdf import PdfReader
import os
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Comm Coach", page_icon="🎙️")

st.title("🎙️ AI Communication Coach")
st.caption("Cloud-Hosted • Mobile Friendly • Multi-Modal")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Keys & Brain")
    
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
    
    # --- INGESTION UI ---
    st.subheader("📚 Add Knowledge")
    
    # 1. YouTube Ingestion
    st.markdown("**Option A: YouTube Video**")
    yt_url = st.text_input("YouTube URL")
    if st.button("Ingest Video"):
        if not yt_url or not pinecone_key or not gemini_key:
            st.error("Missing keys or URL.")
        else:
            with st.spinner("Processing video..."):
                try:
                    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", yt_url).group(1)
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    full_text = " ".join([t['text'] for t in transcript])
                    
                    # Optimization: Chunking with Overlap
                    chunk_size = 1000
                    overlap = 200
                    chunks = []
                    for i in range(0, len(full_text), chunk_size - overlap):
                        chunks.append(full_text[i:i + chunk_size])
                    
                    # Store
                    genai.configure(api_key=gemini_key)
                    pc = Pinecone(api_key=pinecone_key)
                    index = pc.Index("coach-memory")
                    
                    vectors = []
                    for i, chunk in enumerate(chunks):
                        embedding = genai.embed_content(model="models/text-embedding-004", content=chunk)['embedding']
                        vectors.append({
                            "id": f"yt_{video_id}_{i}",
                            "values": embedding,
                            "metadata": {"text": chunk, "source": yt_url, "type": "video"}
                        })
                    
                    index.upsert(vectors)
                    st.success(f"Memorized {len(chunks)} video segments!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # 2. File Ingestion (PDF/TXT)
    st.markdown("**Option B: Upload Files (PDF/TXT)**")
    uploaded_files = st.file_uploader("Upload study materials", accept_multiple_files=True, type=['pdf', 'txt'])
    
    if st.button("Ingest Files") and uploaded_files:
        if not pinecone_key or not gemini_key:
            st.error("Please provide API Keys first.")
        else:
            genai.configure(api_key=gemini_key)
            pc = Pinecone(api_key=pinecone_key)
            index = pc.Index("coach-memory")
            
            progress_bar = st.progress(0)
            
            for file_idx, uploaded_file in enumerate(uploaded_files):
                text_content = ""
                
                # EXTRACT TEXT
                if uploaded_file.type == "application/pdf":
                    reader = PdfReader(uploaded_file)
                    for page in reader.pages:
                        text_content += page.extract_text() + "\n"
                else: # Text file
                    text_content = uploaded_file.read().decode("utf-8")
                
                # CLEAN TEXT (Basic optimization)
                # Remove excessive newlines that break sentences
                text_content = re.sub(r'\n+', ' ', text_content).strip()
                
                # CHUNK WITH OVERLAP (Crucial for context)
                chunk_size = 1000
                overlap = 200
                chunks = []
                for i in range(0, len(text_content), chunk_size - overlap):
                    chunks.append(text_content[i:i + chunk_size])
                
                # EMBED & UPSERT
                vectors = []
                for i, chunk in enumerate(chunks):
                    try:
                        embedding = genai.embed_content(model="models/text-embedding-004", content=chunk)['embedding']
                        vectors.append({
                            "id": f"doc_{uploaded_file.name}_{i}",
                            "values": embedding,
                            "metadata": {"text": chunk, "source": uploaded_file.name, "type": "document"}
                        })
                    except Exception as e:
                        print(f"Skipping chunk {i} due to error: {e}")

                if vectors:
                    index.upsert(vectors)
                
                # Update progress
                progress_bar.progress((file_idx + 1) / len(uploaded_files))
            
            st.success("All files successfully memorized!")

# --- MAIN APP LOGIC ---

if gemini_key and deepgram_key and pinecone_key:
    
    genai.configure(api_key=gemini_key)
    dg_client = DeepgramClient(deepgram_key) 
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index("coach-memory")

    mode = st.radio("Select Mode:", ["🧠 Coach Mode (Advice)", "🎭 Practice Mode (Roleplay)"], horizontal=True)

    audio_bytes = audio_recorder(
        text="Click to Record",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x",
    )

    if audio_bytes:
        with st.spinner("Listening..."):
            payload = {"buffer": audio_bytes}
            options = {"model": "nova-2", "smart_format": True}
            
            response = dg_client.listen.rest.v("1").transcribe_file(payload, options)
            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
            
            if not transcript:
                st.warning("I didn't hear anything. Try again.")
            else:
                st.info(f"You said: {transcript}")

                ai_text_response = ""
                
                if "Coach" in mode:
                    # RAG Retrieval
                    query_embedding = genai.embed_content(
                        model="models/text-embedding-004",
                        content=transcript
                    )['embedding']
                    
                    search_results = index.query(
                        vector=query_embedding,
                        top_k=3,
                        include_metadata=True
                    )
                    
                    context_text = "\n\n".join([
                        f"[Source: {match['metadata']['source']}]\n{match['metadata']['text']}" 
                        for match in search_results['matches']
                    ])
                    
                    prompt = f"""
                    You are an expert communication coach. 
                    USER QUESTION: {transcript}
                    
                    RELEVANT KNOWLEDGE:
                    {context_text}
                    
                    ADVICE: Provide succinct advice based on the knowledge above. Cite the source if possible.
                    """
                    
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

                with st.spinner("Speaking..."):
                    options = {"model": "aura-asteria-en", "text": ai_text_response}
                    # Deepgram TTS typically returns audio bytes, we need to save or stream them.
                    # This save method writes to file system, which works on Streamlit Cloud.
                    filename = "output_audio.mp3"
                    dg_client.speak.rest.v("1").save(filename, options)
                    st.audio(filename, format="audio/mp3", autoplay=True)

else:
    st.warning("👉 Enter your API keys in the sidebar to start.")
