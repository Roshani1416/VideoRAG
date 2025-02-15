from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
import os
import multiprocessing

# Paths
FRAMES_FOLDER = "data/frames_output"
OUTPUT_JSON_PATH = "data/captions.json"
MODEL_PATH = "C:/Users/rosha/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base"

def load_model():
    """Loads the BLIP model and processor."""
    print("🔹 Loading BLIP model... (This may take a few seconds)")
    processor = BlipProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)
    print("✅ Model loaded successfully!")
    return processor, model

def generate_caption(frame_path, processor, model):
    """Generates a caption for a given image."""
    try:
        image = Image.open(frame_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        caption = model.generate(**inputs)
        return processor.decode(caption[0], skip_special_tokens=True)
    except Exception as e:
        print(f"⚠️ Error processing {frame_path}: {e}")
        return None

def process_frame(frame, model_path):
    """Loads model inside worker process and generates a caption."""
    processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

    frame_path = os.path.join(FRAMES_FOLDER, frame)
    if os.path.isfile(frame_path) and frame.lower().endswith((".jpg", ".png")):
        caption = generate_caption(frame_path, processor, model)
        return frame, caption
    return None

if __name__ == '__main__':
    # ✅ Load model once in the main process
    processor, model = load_model()

    # ✅ Ensure directories exist
    os.makedirs(FRAMES_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    frames = os.listdir(FRAMES_FOLDER)
    print(f"🔹 Processing {len(frames)} frames with BLIP...")

    # ✅ Fix multiprocessing on Windows
    multiprocessing.set_start_method('spawn', force=True)

    # ✅ Use multiprocessing for faster captioning
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        captions_list = pool.starmap(process_frame, [(frame, MODEL_PATH) for frame in frames])

    # ✅ Convert list to dictionary (removing None values)
    captions = {frame: caption for frame, caption in captions_list if caption is not None}

    # ✅ Save captions to JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=4)

    print(f"✅ Captions generated successfully! Saved at {OUTPUT_JSON_PATH}")
