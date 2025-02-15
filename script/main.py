import os
from extract_frames import extract_frames
from extract_audio import extract_audio
from transcribe_audio import whisper

# Ensure FFmpeg is in PATH
os.environ["PATH"] += os.pathsep + r"C:\Users\rosha\Downloads\ffmpeg-7.1\bin"

# Define paths
video_path = "data/video.mp4"
output_folder = "data/frames_output"
audio_path = "data/video_audio.mp3"

# Run all steps
extract_frames(video_path, output_folder)
extract_audio(video_path, audio_path)

# Load Whisper and transcribe
model = whisper.load_model("base")
result = model.transcribe(audio_path)

# Save transcription
with open("outputs/transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("✅ Video processing completed!")
