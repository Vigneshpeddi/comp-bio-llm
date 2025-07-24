import gradio as gr
from model import CompBioLLM
import json

def load_model():
    model = CompBioLLM()
    try:
        model.load_fine_tuned_model()
        return model
    except:
        return None

def answer_question(question, temperature=0.7):
    if not question.strip():
        return "Please enter a question about computational biology."
    
    model = load_model()
    if model is None:
        return "Model not found. Please train the model first using train.py"
    
    try:
        answer = model.generate_answer(question, temperature=temperature)
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def create_demo():
    with gr.Blocks(title="Computational Biology Q&A", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Computational Biology Q&A System")
        gr.Markdown("Ask questions about computational biology concepts and get AI-powered answers!")
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g., What is DNA replication?",
                    lines=3
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature (Creativity)",
                    info="Lower values = more focused, Higher values = more creative"
                )
                
                submit_btn = gr.Button("Get Answer", variant="primary")
            
            with gr.Column(scale=2):
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=8,
                    interactive=False
                )
        
        gr.Markdown("### Example Questions:")
        examples = [
            "What is a phylogenetic tree?",
            "What is DNA methylation?",
            "What is the BLAST algorithm?",
            "What is a transcription factor?",
            "What is the central dogma of molecular biology?"
        ]
        
        gr.Examples(
            examples=examples,
            inputs=question_input
        )
        
        gr.Markdown("---")
        gr.Markdown("""
        ### About this System
        This Q&A system is fine-tuned on computational biology concepts using a dataset of questions and answers.
        The model can answer questions about:
        - Molecular biology concepts
        - Bioinformatics algorithms
        - Systems biology
        - Machine learning in biology
        - And much more!
        """)
        
        submit_btn.click(
            fn=answer_question,
            inputs=[question_input, temperature_slider],
            outputs=answer_output
        )
        
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, temperature_slider],
            outputs=answer_output
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True) 