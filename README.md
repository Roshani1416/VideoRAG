# 📌 VideoRAG: Video-Based Retrieval-Augmented Generation

## 📖 Overview
VideoRAG is a Retrieval-Augmented Generation (RAG) system that processes video content by generating captions and answering user queries based on extracted text. It leverages **Whisper**, **sentence-transformers**, and **Qdrant** to store and retrieve relevant information from videos efficiently.

## ✨ Features
- **Automatic Captioning:** Uses OpenAI’s **Whisper** model to generate subtitles for videos.
- **Semantic Search & Retrieval:** Stores extracted text embeddings in **Qdrant** for efficient retrieval.
- **Answer Generation:** Leverages **transformers** to generate answers based on retrieved context.
- **User-Friendly Interface:** Built using **Streamlit** for easy interaction.

## 📂 Project Structure
```
VideoRAG-Project/
│── data/                      # Folder for storing input videos and related data  
│── outputs/                   # Folder for storing generated outputs (captions, text, answers)  
│── script/                    # All Python scripts for processing  
│   │── main.py                # Main script to run the VideoRAG system  
│   │── generate_captions.py    # Generates captions from video using Whisper  
│   │── store_embedding.py       # Converts text into embeddings using sentence-transformers  
│   │── retrieval.py            # Handles RAG-based retrieval using Qdrant  
│   │── answer_generation.py     # Generates answers using transformers  
│   │── app.py                   # Streamlit-based user interface for interaction
│   │── search_video.py          # Search video using CLIP or text queries
|   │──extract_frames.py         # Extracts frames from video
|   │── extract_audio.py          # Converts video to audio
|   │── transcribe_audio.py       # Uses Whisper for speech-to-text
│── venv/                      # Virtual environment (not included in repo)  
│── requirements.txt           # Required dependencies  
│── README.md                  # Project documentation  
│── .gitignore                 # Ignore unnecessary files (e.g., venv, dataset caches)  
```

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/VideoRAG.git  
cd VideoRAG
```

### 2️⃣ Create & Activate a Virtual Environment
```bash
python -m venv venv  
source venv/bin/activate  # On macOS/Linux  
venv\Scripts\activate     # On Windows  
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt  
```

## 🔥 Usage

### Run the Streamlit UI:
```bash
streamlit run script/ui.py  
```

### Run the Main Processing Script:
```bash
python script/main.py --video_path "data/sample.mp4"
```

## 🛠 Dependencies (`requirements.txt`)
```
qdrant-client  
streamlit  
answer-generation  
sentence-transformers  
transformers  
torch  
openai-whisper  
```

## 📌 Future Improvements
- 🔹 Integrate **multi-modal models** for richer video analysis.  
- 🔹 Enhance retrieval accuracy using **fine-tuned embeddings**.  
- 🔹 Deploy as a **web-based application** for easier access.  

## 📜 License
This project is licensed under the MIT License.

---
### 🤝 Contributing
Contributions are welcome! Feel free to submit a PR or open an issue. 😊  

---
🚀 **Developed with ❤️ by [Roshani Singh]**

