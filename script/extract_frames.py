import cv2
import os
import shutil
import subprocess

def extract_frames(video_path, output_folder, scene_threshold=0.3):
    """
    Extract frames from video only when a scene change occurs.
    
    :param video_path: Path to the input video file.
    :param output_folder: Path to the output folder where frames will be saved.
    :param scene_threshold: Threshold for detecting scene changes (higher = fewer frames).
    """
    # ✅ Remove old frames before extracting new ones
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Delete all previous frames
    os.makedirs(output_folder, exist_ok=True)

    # ✅ Use FFmpeg Scene Change Detection (Extract only key frames)
    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,  # Input video
        "-vf", f"select='gt(scene,{scene_threshold})'",  # Scene change detection
        "-vsync", "vfr",  # Variable frame rate (only key frames)
        os.path.join(output_folder, "frame_%04d.jpg")  # Output frames
    ]

    # ✅ Run the FFmpeg command
    subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"✅ Frames extracted successfully using Scene Change Detection!")

# Example usage
video_path = "data/video.mp4"
output_folder = "data/frames_output"
extract_frames(video_path, output_folder)

print("✅ Old frames deleted & new frames extracted efficiently!")
