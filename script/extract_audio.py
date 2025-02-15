import os

def extract_audio(video_path, audio_path):
    cmd = f'ffmpeg -i "{video_path}" -vn -acodec mp3 "{audio_path}"'
    os.system(cmd)

# Run the function
video_path = "data/video.mp4"
audio_path = "data/video_audio.mp3"
extract_audio(video_path, audio_path)
print("✅ Audio extracted successfully!")
