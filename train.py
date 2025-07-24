import os
from data_loader import load_qa_dataset, prepare_dataset
from model import CompBioLLM
from evaluation import ModelEvaluator
import json

def main():
    print("Loading computational biology dataset...")
    data = load_qa_dataset("qa_dataset_expanded.jsonl")
    print(f"Loaded {len(data)} Q&A pairs")
    
    print("Preparing train/test split...")
    train_dataset, test_dataset = prepare_dataset(data, test_size=0.2)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    print("Initializing model...")
    model = CompBioLLM()
    
    print("Starting fine-tuning...")
    model.fine_tune(train_dataset, test_dataset, epochs=3, batch_size=4)
    
    print("Loading fine-tuned model...")
    model.load_fine_tuned_model()
    
    print("Evaluating model performance...")
    evaluator = ModelEvaluator()
    summary, detailed_results = evaluator.evaluate_batch(model, test_dataset, max_samples=10)
    
    print("\n=== EVALUATION RESULTS ===")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")
    
    with open("evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nTraining completed! Model saved to ./comp_bio_model/")
    print("Evaluation results saved to evaluation_results.json")

if __name__ == "__main__":
    main() 