import os
from data_loader import load_qa_dataset, prepare_dataset
from model import CompBioLLM
from evaluation import ModelEvaluator
import json

def quick_train():
    print("Quick Training - Computational Biology Q&A Model")
    print("=" * 60)
    
    print("Loading computational biology dataset...")
    data = load_qa_dataset("qa_dataset_expanded.jsonl")
    print(f"✓ Loaded {len(data)} Q&A pairs")
    
    print("Preparing train/test split...")
    train_dataset, test_dataset = prepare_dataset(data, test_size=0.2)
    print(f"✓ Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    print("Initializing model...")
    model = CompBioLLM()
    print(f"✓ Model: {model.model_name}")
    print(f"✓ Device: {model.device}")
    
    print("Starting quick fine-tuning (1 epoch)...")
    model.fine_tune(train_dataset, test_dataset, epochs=1, batch_size=2)
    print("✓ Fine-tuning completed!")
    
    print("Loading fine-tuned model...")
    model.load_fine_tuned_model()
    print("✓ Model loaded!")
    
    print("Quick evaluation...")
    evaluator = ModelEvaluator()
    summary, detailed_results = evaluator.evaluate_batch(model, test_dataset, max_samples=5)
    
    print("\n=== QUICK EVALUATION RESULTS ===")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")
    
    with open("quick_evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✓ Quick training completed!")
    print("✓ Model saved to ./comp_bio_model/")
    print("✓ Evaluation results saved to quick_evaluation_results.json")

if __name__ == "__main__":
    quick_train() 