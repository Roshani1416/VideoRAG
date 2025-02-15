from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 🔹 Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Connect to Qdrant
qdrant = QdrantClient("http://localhost:6333")

# 🔹 Query Processing
query = "What are neural networks?"
query_embedding = model.encode(query).tolist()

# 🔹 Perform Search
search_results = qdrant.search(
    collection_name="video_rag",
    query_vector=query_embedding,
    limit=5  # ✅ Use `limit` instead of `top` for compatibility
)

# 🔹 Print Search Results
for result in search_results:
    print(f"Frame ID: {result.id}, Score: {result.score}")

print("✅ Search completed!")
