import streamlit as st
import os
import json
from main import extract_frames, extract_audio, whisper
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from answer_generation import generate_answer
from qdrant_client.models import PointStruct
# ✅ Load Embedding Model & Qdrant Client
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient("http://localhost:6333")
COLLECTION_NAME = "video_rag"

# ✅ Streamlit UI
st.title("📽️ VideoRAG - AI-Powered Video Q&A")
captions = {}
# ✅ File Upload
uploaded_file = st.file_uploader("📂 Upload a video", type=["mp4", "mov"])
if uploaded_file:
    st.write("🔄 Processing Video... (This may take a few seconds)")
    
    # Save uploaded file
    video_path = f"data/{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    # ✅ Extract Frames & Audio
    extract_frames(video_path, "data/frames_output")  
    extract_audio(video_path, "data/video_audio.mp3")

    # ✅ Transcribe Audio
    st.write("🔊 Extracting Audio & Transcribing...")
    model = whisper.load_model("base")
    result = model.transcribe("data/video_audio.mp3")
    
    # ✅ Save Transcription
    transcription_path = "outputs/transcription.txt"
    with open(transcription_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    st.write("✅ **Transcription:**", result["text"])

    # ✅ Generate & Store Embeddings
    st.write("📌 **Generating Embeddings & Storing in Qdrant...**")

    # Load or create captions JSON
    captions_json = "data/captions.json"
    captions = {f"frame_{i}.jpg": result["text"] for i in range(5)}  # Temporary captions

    # Save captions
    with open(captions_json, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=4)

    # ✅ Upload to Qdrant
    points = []
    points = []
for idx, (frame, text) in enumerate(captions.items()):
    embedding = embed_model.encode(text).tolist()
    points.append(
        PointStruct(id=idx, vector=embedding, payload={"frame": frame, "text": text})  # ✅ Correct
    )


# ✅ Question Answering Section
st.subheader("🤔 Ask a question about the video")
query = st.text_input("🔍 Type your question here:")

if query:
    st.write("🔹 Searching the video for relevant information...")

    # ✅ Generate Query Embedding
    query_embedding = embed_model.encode(query).tolist()

    # ✅ Search Qdrant
    search_results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=1)

    if search_results:
        best_match = search_results[0]
        retrieved_frame = best_match.id
        retrieved_text = captions.get(f"frame_{retrieved_frame}.jpg", "No relevant caption found.")

        # ✅ Generate Answer using FLAN-T5
        answer = generate_answer(query, retrieved_text)

        # ✅ Display Answer & Frame
        st.subheader("💡 AI Answer:")
        st.write(answer)

    else:
        st.error("❌ No relevant content found in the video. Try another question!")
