from model import CompBioLLM
from data_loader import load_qa_dataset
import json

def test_model():
    print("Testing Computational Biology Q&A Model")
    print("=" * 50)
    
    model = CompBioLLM()
    print(f"✓ Model initialized: {model.model_name}")
    print(f"✓ Device: {model.device}")
    
    data = load_qa_dataset("qa_dataset_expanded.jsonl")
    print(f"✓ Dataset loaded: {len(data)} Q&A pairs")
    
    test_questions = [
        "What is DNA replication?",
        "What is a phylogenetic tree?",
        "What is the BLAST algorithm?"
    ]
    
    print("\nTesting question generation:")
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        try:
            answer = model.generate_answer(question, max_length=100, temperature=0.7)
            print(f"   Answer: {answer}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n✓ Model test completed!")

if __name__ == "__main__":
    test_model() 