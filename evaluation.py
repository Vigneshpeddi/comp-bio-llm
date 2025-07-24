import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import evaluate

class ModelEvaluator:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def semantic_similarity(self, predicted, reference):
        pred_embedding = self.sentence_model.encode([predicted])
        ref_embedding = self.sentence_model.encode([reference])
        similarity = cosine_similarity(pred_embedding, ref_embedding)[0][0]
        return similarity
    
    def rouge_scores(self, predicted, reference):
        scores = self.rouge_scorer.score(reference, predicted)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def bleu_score(self, predicted, reference):
        pred_tokens = word_tokenize(predicted.lower())
        ref_tokens = word_tokenize(reference.lower())
        return sentence_bleu([ref_tokens], pred_tokens)
    
    def exact_match(self, predicted, reference):
        return predicted.strip().lower() == reference.strip().lower()
    
    def evaluate_batch(self, model, test_dataset, max_samples=None):
        predictions = []
        references = []
        
        if max_samples:
            test_data = test_dataset.select(range(min(max_samples, len(test_dataset))))
        else:
            test_data = test_dataset
        
        for i, example in enumerate(test_data):
            question = example['question']
            reference_answer = example['answer']
            
            predicted_answer = model.generate_answer(question)
            
            predictions.append(predicted_answer)
            references.append(reference_answer)
        
        results = {
            'semantic_similarity': [],
            'rouge_scores': {'rouge1': [], 'rouge2': [], 'rougeL': []},
            'bleu_scores': [],
            'exact_matches': []
        }
        
        for pred, ref in zip(predictions, references):
            results['semantic_similarity'].append(self.semantic_similarity(pred, ref))
            rouge_scores = self.rouge_scores(pred, ref)
            for key in rouge_scores:
                results['rouge_scores'][key].append(rouge_scores[key])
            results['bleu_scores'].append(self.bleu_score(pred, ref))
            results['exact_matches'].append(self.exact_match(pred, ref))
        
        summary = {
            'semantic_similarity_mean': float(np.mean(results['semantic_similarity'])),
            'rouge1_mean': float(np.mean(results['rouge_scores']['rouge1'])),
            'rouge2_mean': float(np.mean(results['rouge_scores']['rouge2'])),
            'rougeL_mean': float(np.mean(results['rouge_scores']['rougeL'])),
            'bleu_mean': float(np.mean(results['bleu_scores'])),
            'exact_match_rate': float(np.mean(results['exact_matches'])),
            'total_samples': len(predictions)
        }
        
        return summary, results 