import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Paths
PDF_DIR = "data/rag_docs"
FAISS_INDEX_DIR = "checkpoints/faiss_index"

# 1. Load PDF
print("📂 Loading PDF...")
loader = PyPDFDirectoryLoader(PDF_DIR)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages.")

# 2. Chunk Text
print("✂️ Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks.")

# 3. Download AI Text Model
print("🧠 Downloading model (takes a minute)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Build & Save Database
print("🏗️ Building FAISS database...")
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local(FAISS_INDEX_DIR)
print(f"🎉 Success! Database saved to {FAISS_INDEX_DIR}")