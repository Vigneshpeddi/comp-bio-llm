# Computational Biology Q&A System

An LLM specifically designed to answer questions about computational biology. This project fine-tunes a language model on pre-generated Q&A pairs covering topics in bioinformatics, molecular biology, systems biology, and machine learning applications in biology.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Web Demo](#web-demo)
- [Dataset](#dataset)

## Features

- **Domain-Specific Fine-tuning**: Model trained specifically on computational biology concepts
- **Comprehensive Evaluation**: Multiple metrics including semantic similarity, ROUGE scores, BLEU, and exact match
- **Web Interface**: Both Gradio and Streamlit demos for easy interaction
- **Flexible Generation**: Adjustable temperature and length parameters
- **Easy Deployment**: Simple setup and training process

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- At least 8GB RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vigneshpeddi/comp-bio-llm.git
   cd comp-bio-llm
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

## Quick Start

### 1. Train the Model

```bash
python train.py
```

This will:
- Load the computational biology dataset
- Split into train/test sets
- Fine-tune the model for 3 epochs
- Evaluate performance
- Save the trained model to `./comp_bio_model/`

### 2. Run the Web Demo

**Option A: Gradio Interface**
```bash
python gradio_demo.py
```

**Option B: Streamlit Interface**
```bash
streamlit run streamlit_demo.py
```

### 3. Test Individual Questions

```python
from model import CompBioLLM

model = CompBioLLM()
model.load_fine_tuned_model()

answer = model.generate_answer("What is DNA replication?")
print(answer)
```

## Usage

### Training Configuration

The model can be configured in `model.py`:

```python
# Model parameters
model_name = "microsoft/DialoGPT-medium"  # Base model
max_length = 512                          # Maximum sequence length
epochs = 3                                # Training epochs
batch_size = 4                            # Batch size
```

### Evaluation

Run evaluation on the test set:

```python
from evaluation import ModelEvaluator
from model import CompBioLLM

model = CompBioLLM()
model.load_fine_tuned_model()

evaluator = ModelEvaluator()
summary, results = evaluator.evaluate_batch(model, test_dataset)
print(summary)
```

### Custom Questions

```python
# Load the model
model = CompBioLLM()
model.load_fine_tuned_model()

# Ask questions
questions = [
    "What is a phylogenetic tree?",
    "How does BLAST work?",
    "What is the central dogma of molecular biology?"
]

for question in questions:
    answer = model.generate_answer(question, temperature=0.7)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

## Model Architecture

- **Base Model**: Microsoft DialoGPT-medium (345M parameters)
- **Fine-tuning**: Causal language modeling with custom prompt format
- **Prompt Format**: `"Question: {question}\nAnswer: {answer}"`
- **Generation**: Temperature-controlled sampling with configurable length

### Training Process

1. **Data Preprocessing**: Q&A pairs formatted into training prompts
2. **Tokenization**: Using model-specific tokenizer with padding
3. **Fine-tuning**: Causal language modeling objective
4. **Evaluation**: Multiple metrics on held-out test set

## Evaluation Metrics

The system evaluates performance using:

- **Semantic Similarity**: Cosine similarity between predicted and reference embeddings
- **ROUGE Scores**: ROUGE-1, ROUGE-2, and ROUGE-L for text overlap
- **BLEU Score**: Bilingual Evaluation Understudy for n-gram overlap
- **Exact Match**: Percentage of exactly matching answers

### Expected Performance

Based on the computational biology dataset:
- Semantic Similarity: ~0.7-0.8
- ROUGE-1: ~0.4-0.5
- Exact Match Rate: ~0.1-0.2

## Web Demo

### Gradio Interface

```bash
python gradio_demo.py
```

### Streamlit Interface

```bash
streamlit run streamlit_demo.py
```

## Dataset

The `qa_dataset_expanded.jsonl` contains 55 pre-generated Q&A pairs covering:

### Topics Covered

- **Molecular Biology**: DNA, RNA, proteins, gene expression
- **Bioinformatics**: Algorithms, sequence analysis, databases
- **Systems Biology**: Networks, pathways, modeling
- **Machine Learning**: Classification, clustering, feature engineering
- **Computational Methods**: Alignment, phylogenetics, structure prediction

### Dataset Format

```json
{
  "question": "What is DNA replication?",
  "answer": "DNA replication is the process of copying the DNA molecule to ensure each new cell receives an identical set of DNA."
}
```

### Adding New Data

To expand the dataset:

1. Add new Q&A pairs to `qa_dataset_expanded.jsonl`
2. Retrain the model: `python train.py`
3. The model will automatically use the updated dataset

---

**Note**: This system is designed for educational and research purposes. For clinical or commercial applications, additional validation and testing is recommended. 