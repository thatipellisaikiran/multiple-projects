from setuptools import setup, find_packages

setup(
    name="rag-quote-retrieval",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "datasets>=2.8.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.12.0",
        "faiss-cpu>=1.7.0",
        "scikit-learn>=1.1.0",
        "streamlit>=1.28.0",
        "plotly>=5.10.0",
        "ragas>=0.0.18",
        "transformers>=4.21.0",
        "huggingface-hub>=0.10.0",
    ],
    author="Your Name",
    description="RAG-based semantic quote retrieval system with fine-tuned embeddings",
    python_requires=">=3.8",
)