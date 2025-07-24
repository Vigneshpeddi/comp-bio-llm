#!/usr/bin/env python3

import subprocess
import sys
import os

def run_command(command, description):
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python 3.8 or higher is required")
        return False
    print(f"Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    print("Installing dependencies...")
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def verify_installation():
    print("Verifying installation...")
    
    try:
        import torch
        print(f"PyTorch {torch.__version__} installed")
        
        import transformers
        print(f"Transformers {transformers.__version__} installed")
        
        import gradio
        print(f"Gradio {gradio.__version__} installed")
        
        import streamlit
        print(f"Streamlit {streamlit.__version__} installed")
        
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available - training will use CPU")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_dataset():
    print("Testing dataset...")
    try:
        from data_loader import load_qa_dataset
        data = load_qa_dataset("qa_dataset_expanded.jsonl")
        print(f"Dataset loaded: {len(data)} Q&A pairs")
        return True
    except Exception as e:
        print(f"Dataset test failed: {e}")
        return False

def main():
    print("Computational Biology Q&A System Setup")
    print("=" * 50)
    
    if not check_python_version():
        sys.exit(1)
    
    if not install_dependencies():
        print("Setup failed during dependency installation")
        sys.exit(1)
    
    if not verify_installation():
        print("Setup failed during verification")
        sys.exit(1)
    
    if not test_dataset():
        print("Setup failed during dataset test")
        sys.exit(1)
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Run quick training: python quick_train.py")
    print("2. Test the model: python test_model.py")
    print("3. Launch web demo: python gradio_demo.py")
    print("4. Or use Streamlit: streamlit run streamlit_demo.py")

if __name__ == "__main__":
    main() 