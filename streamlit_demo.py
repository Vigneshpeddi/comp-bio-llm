import streamlit as st
from model import CompBioLLM
import json

@st.cache_resource
def load_model():
    model = CompBioLLM()
    try:
        model.load_fine_tuned_model()
        return model
    except:
        return None

def main():
    st.set_page_config(
        page_title="Computational Biology Q&A",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("Computational Biology Q&A System")
    st.markdown("Ask questions about computational biology concepts and get AI-powered answers!")
    
    model = load_model()
    
    if model is None:
        st.error("Model not found. Please train the model first using `python train.py`")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question = st.text_area(
            "Question",
            placeholder="e.g., What is DNA replication?",
            height=100
        )
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            temperature = st.slider(
                "Temperature (Creativity)",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Lower values = more focused, Higher values = more creative"
            )
        
        with col1_2:
            max_length = st.slider(
                "Max Answer Length",
                min_value=50,
                max_value=500,
                value=200,
                step=50
            )
        
        if st.button("Get Answer", type="primary"):
            if question.strip():
                with st.spinner("Generating answer..."):
                    try:
                        answer = model.generate_answer(question, max_length=max_length, temperature=temperature)
                        st.success("Answer generated successfully!")
                        st.text_area("Answer", answer, height=200, disabled=True)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.markdown("### Example Questions")
        examples = [
            "What is a phylogenetic tree?",
            "What is DNA methylation?",
            "What is the BLAST algorithm?",
            "What is a transcription factor?",
            "What is the central dogma of molecular biology?",
            "What is a k-mer?",
            "What is sequence alignment?",
            "What is gene expression?"
        ]
        
        for example in examples:
            if st.button(example, key=example):
                st.session_state.question = example
                st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This Q&A system is fine-tuned on computational biology concepts using a dataset of questions and answers.
        
        **Topics covered:**
        - Molecular biology
        - Bioinformatics algorithms
        - Systems biology
        - Machine learning in biology
        - And much more!
        """)
    
    if 'question' in st.session_state:
        st.text_area("Question (from example)", st.session_state.question, height=100, disabled=True)
        if st.button("Generate Answer for Example"):
            with st.spinner("Generating answer..."):
                try:
                    answer = model.generate_answer(st.session_state.question, max_length=max_length, temperature=temperature)
                    st.success("Answer generated successfully!")
                    st.text_area("Answer", answer, height=200, disabled=True)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main() 