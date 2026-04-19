import os
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Suppress warnings for a clean terminal
warnings.filterwarnings("ignore")

# --- THE ULTIMATE OVERRIDE ---
# Paste your brand new key right inside these quotes!
MY_API_KEY = "AIzaSyAVXjLkSdehGBmshX-9knjYO1cw6Z61sqg"
# ------------------------------

FAISS_INDEX_DIR = "checkpoints/faiss_index"

print("🧠 1. Waking up the Gemini 1.5 Brain...")
# Explicitly force the model to use the hardcoded key
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=MY_API_KEY)

print("📚 2. Loading the Agricultural Memory (FAISS)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# Define how we want the AI to behave
prompt_template = """
You are Agri-AI. Use the context to answer the question.
CRITICAL RULE: Give answers point-wise. Use very short points and words. Be direct.
If it is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

print("🔗 3. Connecting Gemini to your local database...")
# This chain searches FAISS, grabs the top 3 paragraphs, and sends them to Gemini
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT}
)

print("\n==================================================")
print("✨ AGRI-AI (POWERED BY GEMINI) IS READY! Type 'quit' to exit.")
print("==================================================\n")

# Interactive chat loop
while True:
    user_question = input("🧑‍🌾 Ask a farming question: ")
    
    if user_question.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
        
    print("🤖 Thinking...")
    try:
        # Ask Gemini!
        result = qa_chain.invoke({"query": user_question})
        print(f"\n🌱 AI Answer: {result['result']}\n")
    except Exception as e:
        print(f"\n❌ Oops, something went wrong: {e}\n")