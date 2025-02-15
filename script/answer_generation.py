from transformers import pipeline
import torch

# ✅ Use GPU if available
device = 0 if torch.cuda.is_available() else -1  

# ✅ Load FLAN-T5 (Best Small Model for QA)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", device=device)

def generate_answer(query, retrieved_text):
    """Generates an AI response based on video transcript."""
    prompt = f"Based on the video transcript:\n{retrieved_text}\n\nAnswer the question: {query}"
    response = qa_pipeline(prompt, max_length=100)[0]["generated_text"]
    return response

# ✅ Test the function (Only runs when executed directly, not when imported)
if __name__ == "__main__":
    test_query = "What are neural networks?"
    test_text = """Neural networks are computational models inspired by the human brain, 
    consisting of layers of interconnected nodes (neurons) that process and learn patterns from data. 
    They are the backbone of deep learning, enabling tasks like image recognition, 
    natural language processing, and autonomous decision-making."""

    test_answer = generate_answer(test_query, test_text)
    print("💡 AI Answer:", test_answer)
