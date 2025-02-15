import os
os.environ["PATH"] += os.pathsep + r"C:\Users\rosha\Downloads\ffmpeg-7.1-essentials_build\ffmpeg-7.1-essentials_build\bin"

import whisper


# Ensure the outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Load Whisper model
model = whisper.load_model("small")

# Transcribe audio
audio_path = "data/video_audio.mp3"
result = model.transcribe(audio_path)

# Save transcription
output_path = "outputs/transcription.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"✅ Transcription completed! File saved at: {output_path}")
