from model import CompBioLLM
from data_loader import load_qa_dataset
from evaluation import ModelEvaluator
import json

def example_usage():
    print("Computational Biology Q&A System - Example Usage")
    print("=" * 60)
    
    print("\n1. Loading and testing the base model...")
    model = CompBioLLM()
    print(f"   Model: {model.model_name}")
    print(f"   Device: {model.device}")
    
    print("\n2. Testing with sample questions...")
    sample_questions = [
        "What is DNA replication?",
        "What is a phylogenetic tree?",
        "What is the BLAST algorithm?",
        "What is the central dogma of molecular biology?",
        "What is a k-mer?"
    ]
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n   Q{i}: {question}")
        try:
            answer = model.generate_answer(question, max_length=150, temperature=0.7)
            print(f"   A{i}: {answer}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n3. Loading dataset for evaluation...")
    data = load_qa_dataset("qa_dataset_expanded.jsonl")
    print(f"   Dataset size: {len(data)} Q&A pairs")
    
    print("\n4. Sample dataset entries...")
    for i, entry in enumerate(data[:3], 1):
        print(f"   Entry {i}:")
        print(f"     Q: {entry['question']}")
        print(f"     A: {entry['answer']}")
    
    print("\n5. Model capabilities...")
    print("   - Answer questions about computational biology")
    print("   - Adjustable creativity (temperature)")
    print("   - Configurable answer length")
    print("   - Support for various bioinformatics topics")
    
    print("\n6. Next steps...")
    print("   - Run 'python quick_train.py' to fine-tune the model")
    print("   - Run 'python gradio_demo.py' for web interface")
    print("   - Run 'streamlit run streamlit_demo.py' for Streamlit demo")
    
    print("\nExample usage completed!")

if __name__ == "__main__":
    example_usage() 