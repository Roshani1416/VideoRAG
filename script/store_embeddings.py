from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import json
import time  # ✅ Added to handle server connection issues

# ✅ Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Connect to Qdrant
try:
    qdrant = QdrantClient("http://localhost:6333")
    qdrant.get_collections()  # ✅ Test connection
    print("✅ Connected to Qdrant!")
except Exception as e:
    print(f"❌ Failed to connect to Qdrant: {e}")
    exit(1)  # Stop execution if Qdrant isn't running

# ✅ Define Collection Name
COLLECTION_NAME = "video_rag"

# ✅ Ensure Collection Exists
collections = qdrant.get_collections()
existing_collections = [col.name for col in collections.collections]

if COLLECTION_NAME not in existing_collections:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"✅ Created collection: {COLLECTION_NAME}")

# ✅ Load Captions
with open("data/captions.json", "r", encoding="utf-8") as f:
    captions = json.load(f)

# ✅ Prepare Data for Qdrant Upload
points = []
for idx, (frame, text) in enumerate(captions.items()):
    embedding = model.encode(text).tolist()

    # ✅ Create PointStruct (Ensure It's Not a Dict)
    point = PointStruct(
        id=idx,  # Ensure ID is an integer
        vector=embedding,
        payload={"frame": frame, "text": text}  # Metadata
    )

    points.append(point)

# ✅ Debugging: Check If Points Are in Correct Format
print(f"🔍 Example PointStruct: {points[0]}")  # Print one sample to verify

# ✅ Upload to Qdrant in Batches
BATCH_SIZE = 50
print(f"🔹 Uploading {len(points)} embeddings to Qdrant in batches of {BATCH_SIZE}...")

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i : i + BATCH_SIZE]  # ✅ Slice batch properly
    
    # ✅ Final Fix: Use `upsert()` Instead of `upload_points()`
    try:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
        time.sleep(0.5)  # ✅ Add slight delay to prevent overload issues
    except Exception as e:
        print(f"❌ Error uploading batch {i // BATCH_SIZE + 1}: {e}")

print("✅ Embeddings stored successfully!")
