import os
import sys
from pathlib import Path

def check_environment():
    print("=== AgriOmni-VLM Environment Verification ===")
    
    # 1. Check Python Version
    print(f"\n[1/4] Python Version: {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("⚠️ Warning: Python 3.11+ is recommended.")

    # 2. Check Core Packages
    print("\n[2/4] Checking Core Packages:")
    packages = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "peft": "PEFT (LoRA)",
        "faster_whisper": "Faster Whisper",
        "faiss": "FAISS",
        "gradio": "Gradio UI"
    }
    
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f"✅ {name} installed successfully.")
        except ImportError:
            print(f"❌ {name} is MISSING. Please run: pip install -r requirements.txt")

    # 3. Check GPU Availability
    print("\n[3/4] Checking Hardware:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️ No GPU detected! PyTorch is using CPU. Training will be extremely slow.")
    except ImportError:
        print("❌ Cannot check GPU; PyTorch not loaded.")

    # 4. Check Directory Structure
    print("\n[4/4] Checking Project Structure:")
    directories = [
        "data/plantvillage", "data/plantdoc", "data/fruit", 
        "data/plantseg", "data/agmmu", "data/rag_docs",
        "src/classification", "src/segmentation", "src/vlm", 
        "src/rag", "src/api",
        "checkpoints", "outputs", "logs", "notebooks"
    ]
    
    missing_dirs = []
    for d in directories:
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            missing_dirs.append(d)
            
    if missing_dirs:
        print(f"✅ Created {len(missing_dirs)} missing directories.")
    else:
        print("✅ All project directories exist.")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    check_environment()