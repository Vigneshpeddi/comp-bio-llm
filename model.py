import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np

class CompBioLLM:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def tokenize_function(self, examples):
        prompts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
        return self.tokenizer(prompts, truncation=True, padding=True, max_length=512)
    
    def fine_tune(self, train_dataset, val_dataset=None, epochs=3, batch_size=4):
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        
        if val_dataset:
            tokenized_val = val_dataset.map(self.tokenize_function, batched=True)
        else:
            tokenized_val = None
        
        training_args = TrainingArguments(
            output_dir="./comp_bio_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=500,
            eval_steps=500 if val_dataset else None,
            save_total_limit=2,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
        )
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained("./comp_bio_model")
    
    def generate_answer(self, question, max_length=200, temperature=0.7):
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip()
        return answer
    
    def load_fine_tuned_model(self, model_path="./comp_bio_model"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device) 