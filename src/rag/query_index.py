import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_DIR = "checkpoints/faiss_index"

print("🧠 Waking up the AI Memory...")

# 1. Load the exact same embedding model we used to build the index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load the database from your hard drive
try:
    # Note: allow_dangerous_deserialization is required for local FAISS files in newer Langchain versions
    vector_db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    print("✅ Memory loaded successfully!")
except Exception as e:
    print(f"❌ Could not load memory. Error: {e}")
    exit()

# 3. Ask a question! 
# (You can change this string to anything you want to ask the PDF)
question = "What are the main benefits of sustainable farming and soil health?"
print(f"\n🗣️ You asked: '{question}'\n")

# 4. Search the database
print("🔍 Searching through the agricultural manuals...\n")
results = vector_db.similarity_search(question, k=3) # Retrieve the top 3 most relevant chunks

print("==================================================")
print("📖 HERE IS WHAT THE AI FOUND IN THE PDF:")
print("==================================================")
for i, doc in enumerate(results):
    # Print the source page number and the text chunk
    page_num = doc.metadata.get('page', 'Unknown')
    print(f"\n--- Excerpt {i+1} (From Page {page_num}) ---")
    print(doc.page_content)
print("\n==================================================")