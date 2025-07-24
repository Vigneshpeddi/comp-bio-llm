import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

def load_qa_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def prepare_dataset(data, test_size=0.2, random_state=42):
    df = pd.DataFrame(data)
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    
    return train_dataset, test_dataset

def format_qa_prompt(question, answer=None):
    if answer:
        return f"Question: {question}\nAnswer: {answer}"
    else:
        return f"Question: {question}\nAnswer:" 