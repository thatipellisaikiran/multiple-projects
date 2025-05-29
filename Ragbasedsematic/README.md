# 📚 RAG-Based Semantic Quote Retrieval System

This project implements a comprehensive **Retrieval Augmented Generation (RAG)** system for semantic quote retrieval using the [`Abirate/english_quotes`](https://huggingface.co/datasets/Abirate/english_quotes) dataset from HuggingFace.

---

## 🎯 Project Overview

The system consists of:

* **Data Preparation**: Loading and preprocessing the `english_quotes` dataset
* **Model Fine-tuning**: Fine-tuning a sentence transformer for better quote matching
* **RAG Pipeline**: Building a semantic search system with FAISS indexing
* **Evaluation**: Comprehensive evaluation using multiple metrics
* **Streamlit App**: Interactive web interface for quote retrieval

---

## 🚀 Features

* 🔎 **Semantic Search**: Find quotes using natural language queries
* 🧠 **Fine-tuned Embeddings**: Custom-trained model for better quote understanding
* 🧹 **Multi-faceted Retrieval**: Search by author, theme, or content
* 🖥️ **Interactive UI**: User-friendly Streamlit interface
* 🧾 **JSON Export**: Download search results in structured format
* 📊 **Analytics Dashboard**: Visualize dataset statistics
* 📈 **Evaluation Metrics**: Comprehensive system performance analysis

---

## 📋 Requirements

* Python 3.8+
* CUDA-compatible GPU (optional, for faster training)
* 8GB+ RAM recommended
* Internet connection (for dataset download)

---

## 🛠️ Installation

### Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Alternative installation:

```bash
pip install -e .
```

---

## 🎮 Usage

### 🌐 Running the Streamlit App

Launch the interactive web interface:

```bash
streamlit run rag_quote_system.py
```

Then open your browser to [http://localhost:8501](http://localhost:8501)

---
