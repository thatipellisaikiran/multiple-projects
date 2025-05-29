# RAG-Based Semantic Quote Retrieval System
# Task 2 - Complete Implementation

import pandas as pd
import numpy as np
import json
import pickle
from typing import List, Dict, Any, Tuple
import re
from datetime import datetime

# Core ML/NLP libraries
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
#import torch
from torch.utils.data import DataLoader
import faiss
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit and UI
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# RAG Evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Warnings
import warnings
warnings.filterwarnings('ignore')

class QuoteDataProcessor:
    """Data preparation and preprocessing for the quotes dataset"""
    
    def __init__(self):
        self.dataset = None
        self.processed_data = None
        
    def load_and_explore_data(self):
        """Load the english_quotes dataset from HuggingFace"""
        print("Loading english_quotes dataset...")
        try:
            self.dataset = load_dataset("Abirate/english_quotes")
            print(f"Dataset loaded successfully!")
            print(f"Train samples: {len(self.dataset['train'])}")
            
            # Display sample data
            sample = self.dataset['train'][0]
            print(f"\nSample quote structure:")
            for key, value in sample.items():
                print(f"{key}: {value}")
                
            return self.dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def preprocess_data(self):
        """Clean and preprocess the dataset"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_and_explore_data() first.")
        
        print("Preprocessing data...")
        train_data = self.dataset['train']
        
        processed_quotes = []
        for i, item in enumerate(train_data):
            try:
                # Clean quote text
                quote = str(item.get('quote', '')).strip()
                author = str(item.get('author', 'Unknown')).strip()
                tags = item.get('tags', [])
                
                # Skip empty quotes
                if not quote or len(quote) < 10:
                    continue
                
                # Clean quote text
                quote_clean = self._clean_text(quote)
                author_clean = self._clean_text(author)
                
                # Process tags
                if isinstance(tags, str):
                    tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
                elif isinstance(tags, list):
                    tags = [str(tag).strip() for tag in tags if str(tag).strip()]
                else:
                    tags = []
                
                processed_quotes.append({
                    'id': i,
                    'quote': quote_clean,
                    'author': author_clean,
                    'tags': tags,
                    'combined_text': f"{quote_clean} by {author_clean} tags: {', '.join(tags)}"
                })
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue
        
        self.processed_data = processed_quotes
        print(f"Processed {len(processed_quotes)} quotes successfully")
        return processed_quotes
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove or replace problematic characters
        text = re.sub(r'[^\w\s\-\.,!?;:\'"()]', ' ', text)
        return text
    
    def create_training_data(self) -> List[InputExample]:
        """Create training examples for sentence transformer fine-tuning"""
        if not self.processed_data:
            raise ValueError("Data not processed. Call preprocess_data() first.")
        
        examples = []
        
        # Create positive pairs (quote-author, quote-tags)
        for item in self.processed_data:
            quote = item['quote']
            author = item['author']
            tags = item['tags']
            
            # Quote-Author pairs
            if author and author != 'Unknown':
                query = f"quotes by {author}"
                examples.append(InputExample(texts=[query, quote], label=1.0))
            
            # Quote-Tag pairs
            for tag in tags:
                if tag:
                    query = f"quotes about {tag}"
                    examples.append(InputExample(texts=[query, quote], label=0.8))
            
            # Combined queries
            if author != 'Unknown' and tags:
                query = f"quotes about {tags[0]} by {author}"
                examples.append(InputExample(texts=[query, quote], label=1.0))
        
        print(f"Created {len(examples)} training examples")
        return examples

class QuoteEmbeddingModel:
    """Fine-tune sentence transformer for quote retrieval"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
    def load_base_model(self):
        """Load the base sentence transformer model"""
        print(f"Loading base model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def fine_tune(self, training_examples: List[InputExample], 
                  epochs: int = 3, batch_size: int = 16):
        """Fine-tune the model on quote data"""
        if self.model is None:
            self.load_base_model()
        
        print(f"Fine-tuning model with {len(training_examples)} examples...")
        
        # Create dataloader
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)
        
        # Define loss
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            show_progress_bar=True
        )
        
        self.is_trained = True
        print("Fine-tuning completed!")
        return self.model
    
    def save_model(self, path: str):
        """Save the fine-tuned model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a fine-tuned model"""
        self.model = SentenceTransformer(path)
        self.is_trained = True
        print(f"Model loaded from {path}")

class RAGPipeline:
    """Retrieval Augmented Generation pipeline for quotes"""
    
    def __init__(self, embedding_model: SentenceTransformer, quotes_data: List[Dict]):
        self.embedding_model = embedding_model
        self.quotes_data = quotes_data
        self.index = None
        self.embeddings = None
        
    def build_index(self):
        """Build FAISS index from quote embeddings"""
        print("Building FAISS index...")
        
        # Create embeddings for all quotes
        texts = []
        for quote in self.quotes_data:
            # Combine quote, author, and tags for better retrieval
            combined = f"{quote['quote']} by {quote['author']}"
            if quote['tags']:
                combined += f" tags: {', '.join(quote['tags'])}"
            texts.append(combined)
        
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Index built with {len(texts)} quotes")
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant quotes for a query"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.quotes_data):
                result = self.quotes_data[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def generate_response(self, query: str, retrieved_quotes: List[Dict]) -> Dict:
        """Generate structured response using retrieved quotes"""
        # For this implementation, we'll create a structured response
        # In a full implementation, you'd use an LLM here
        
        response = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'retrieved_quotes': [],
            'summary': self._create_summary(query, retrieved_quotes),
            'total_results': len(retrieved_quotes)
        }
        
        for quote_data in retrieved_quotes:
            response['retrieved_quotes'].append({
                'quote': quote_data['quote'],
                'author': quote_data['author'],
                'tags': quote_data['tags'],
                'similarity_score': quote_data.get('similarity_score', 0.0)
            })
        
        return response
    
    def _create_summary(self, query: str, quotes: List[Dict]) -> str:
        """Create a summary of retrieved quotes"""
        if not quotes:
            return "No relevant quotes found for your query."
        
        authors = list(set([q['author'] for q in quotes if q['author'] != 'Unknown']))
        all_tags = []
        for q in quotes:
            all_tags.extend(q['tags'])
        common_tags = list(set(all_tags))[:5]
        
        summary = f"Found {len(quotes)} relevant quotes"
        if authors:
            summary += f" from authors including {', '.join(authors[:3])}"
        if common_tags:
            summary += f" related to themes: {', '.join(common_tags)}"
        
        return summary

class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        
    def create_evaluation_dataset(self) -> List[Dict]:
        """Create evaluation queries and expected results"""
        eval_queries = [
            {
                'query': 'quotes about love',
                'expected_themes': ['love', 'romance', 'heart']
            },
            {
                'query': 'motivational quotes about success',
                'expected_themes': ['success', 'achievement', 'motivation']
            },
            {
                'query': 'quotes by Oscar Wilde',
                'expected_author': 'Oscar Wilde'
            },
            {
                'query': 'quotes about life and wisdom',
                'expected_themes': ['life', 'wisdom', 'philosophy']
            },
            {
                'query': 'inspirational quotes about courage',
                'expected_themes': ['courage', 'brave', 'fear']
            }
        ]
        
        return eval_queries
    
    def evaluate_retrieval_quality(self, eval_queries: List[Dict]) -> Dict:
        """Evaluate retrieval quality"""
        results = {
            'query_results': [],
            'average_relevance': 0.0,
            'author_accuracy': 0.0,
            'theme_relevance': 0.0
        }
        
        total_relevance = 0
        author_matches = 0
        theme_matches = 0
        total_author_queries = 0
        total_theme_queries = 0
        
        for eval_item in eval_queries:
            query = eval_item['query']
            retrieved = self.rag_pipeline.retrieve(query, top_k=5)
            
            # Calculate relevance based on similarity scores
            avg_similarity = np.mean([r.get('similarity_score', 0) for r in retrieved])
            total_relevance += avg_similarity
            
            # Check author accuracy
            if 'expected_author' in eval_item:
                total_author_queries += 1
                expected_author = eval_item['expected_author'].lower()
                for quote in retrieved:
                    if expected_author in quote['author'].lower():
                        author_matches += 1
                        break
            
            # Check theme relevance
            if 'expected_themes' in eval_item:
                total_theme_queries += 1
                expected_themes = [t.lower() for t in eval_item['expected_themes']]
                found_themes = False
                for quote in retrieved:
                    quote_text = (quote['quote'] + ' ' + ' '.join(quote['tags'])).lower()
                    if any(theme in quote_text for theme in expected_themes):
                        found_themes = True
                        break
                if found_themes:
                    theme_matches += 1
            
            results['query_results'].append({
                'query': query,
                'num_results': len(retrieved),
                'avg_similarity': avg_similarity,
                'top_result': retrieved[0] if retrieved else None
            })
        
        results['average_relevance'] = total_relevance / len(eval_queries)
        results['author_accuracy'] = author_matches / total_author_queries if total_author_queries > 0 else 0
        results['theme_relevance'] = theme_matches / total_theme_queries if total_theme_queries > 0 else 0
        
        return results

def create_streamlit_app():
    """Create Streamlit application for quote retrieval"""
    
    st.set_page_config(
        page_title="RAG Quote Retrieval System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG-Based Quote Retrieval System")
    st.markdown("Find inspiring quotes using semantic search powered by fine-tuned embeddings!")
    
    # Initialize session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
        st.session_state.quotes_data = None
    
    # Sidebar for system status
    with st.sidebar:
        st.header("System Status")
        
        if st.button("Initialize System"):
            with st.spinner("Loading and processing data..."):
                try:
                    # Load and process data
                    processor = QuoteDataProcessor()
                    processor.load_and_explore_data()
                    quotes_data = processor.preprocess_data()
                    
                    # Load model (use base model for demo)
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Create RAG pipeline
                    rag_pipeline = RAGPipeline(model, quotes_data)
                    rag_pipeline.build_index()
                    
                    st.session_state.rag_pipeline = rag_pipeline
                    st.session_state.quotes_data = quotes_data
                    
                    st.success("System initialized successfully!")
                    st.info(f"Loaded {len(quotes_data)} quotes")
                    
                except Exception as e:
                    st.error(f"Error initializing system: {e}")
        
        if st.session_state.rag_pipeline:
            st.success("✅ System Ready")
            st.info(f"📊 {len(st.session_state.quotes_data)} quotes indexed")
    
    # Main interface
    if st.session_state.rag_pipeline:
        
        # Query input
        st.header("🔍 Search Quotes")
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., 'motivational quotes about success' or 'quotes by Oscar Wilde'"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            num_results = st.slider("Number of results:", 1, 10, 5)
        with col2:
            search_button = st.button("Search", type="primary")
        
        if query and search_button:
            with st.spinner("Searching for relevant quotes..."):
                try:
                    # Retrieve quotes
                    retrieved_quotes = st.session_state.rag_pipeline.retrieve(query, top_k=num_results)
                    response = st.session_state.rag_pipeline.generate_response(query, retrieved_quotes)
                    
                    # Display results
                    st.header("📋 Results")
                    
                    # Summary
                    st.info(response['summary'])
                    
                    # Individual quotes
                    for i, quote_data in enumerate(response['retrieved_quotes'], 1):
                        with st.expander(f"Quote {i} (Similarity: {quote_data['similarity_score']:.3f})"):
                            st.markdown(f"**Quote:** _{quote_data['quote']}_")
                            st.markdown(f"**Author:** {quote_data['author']}")
                            if quote_data['tags']:
                                st.markdown(f"**Tags:** {', '.join(quote_data['tags'])}")
                            
                            # Progress bar for similarity score
                            st.progress(quote_data['similarity_score'])
                    
                    # JSON export
                    st.header("📄 Export Results")
                    json_str = json.dumps(response, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"quote_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    # Display JSON (collapsible)
                    with st.expander("View JSON Response"):
                        st.json(response)
                
                except Exception as e:
                    st.error(f"Error during search: {e}")
        
        # Analytics section
        if st.session_state.quotes_data:
            st.header("📊 Dataset Analytics")
            
            # Author distribution
            authors = [q['author'] for q in st.session_state.quotes_data if q['author'] != 'Unknown']
            author_counts = pd.Series(authors).value_counts().head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=author_counts.values,
                    y=author_counts.index,
                    orientation='h',
                    title="Top 10 Authors by Quote Count"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Tag distribution
                all_tags = []
                for quote in st.session_state.quotes_data:
                    all_tags.extend(quote['tags'])
                
                if all_tags:
                    tag_counts = pd.Series(all_tags).value_counts().head(10)
                    fig = px.pie(
                        values=tag_counts.values,
                        names=tag_counts.index,
                        title="Top 10 Tags Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Please initialize the system using the sidebar button.")
        st.markdown("""
        ### About This System
        
        This RAG-based quote retrieval system:
        
        1. **Loads** the english_quotes dataset from HuggingFace
        2. **Fine-tunes** a sentence transformer model for better quote matching
        3. **Builds** a semantic search index using FAISS
        4. **Retrieves** relevant quotes based on natural language queries
        5. **Provides** structured JSON responses with similarity scores
        
        **Example queries to try:**
        - "motivational quotes about success"
        - "quotes by famous authors about love"
        - "inspirational quotes about courage and strength"
        - "philosophical quotes about life and death"
        """)

def main():
    """Main execution function for the complete RAG system"""
    
    print("=== RAG-Based Quote Retrieval System ===")
    print("Task 2 Implementation\n")
    
    # Step 1: Data Preparation
    print("1. Data Preparation")
    processor = QuoteDataProcessor()
    dataset = processor.load_and_explore_data()
    
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return
    
    quotes_data = processor.preprocess_data()
    training_examples = processor.create_training_data()
    
    # Step 2: Model Fine-tuning
    print("\n2. Model Fine-tuning")
    embedding_model = QuoteEmbeddingModel()
    embedding_model.load_base_model()
    
    # For demonstration, we'll use a smaller subset for fine-tuning
    sample_examples = training_examples[:1000] if len(training_examples) > 1000 else training_examples
    embedding_model.fine_tune(sample_examples, epochs=1, batch_size=16)
    
    # Save the model
    model_path = "fine_tuned_quote_model"
    embedding_model.save_model(model_path)
    
    # Step 3: Build RAG Pipeline
    print("\n3. Building RAG Pipeline")
    rag_pipeline = RAGPipeline(embedding_model.model, quotes_data)
    rag_pipeline.build_index()
    
    # Step 4: RAG Evaluation
    print("\n4. RAG Evaluation")
    evaluator = RAGEvaluator(rag_pipeline)
    eval_queries = evaluator.create_evaluation_dataset()
    evaluation_results = evaluator.evaluate_retrieval_quality(eval_queries)
    
    print("Evaluation Results:")
    print(f"Average Relevance: {evaluation_results['average_relevance']:.3f}")
    print(f"Author Accuracy: {evaluation_results['author_accuracy']:.3f}")
    print(f"Theme Relevance: {evaluation_results['theme_relevance']:.3f}")
    
    # Test queries
    print("\n5. Testing Sample Queries")
    test_queries = [
        "quotes about love and relationships",
        "motivational quotes by famous authors",
        "quotes about success and achievement"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = rag_pipeline.retrieve(query, top_k=3)
        response = rag_pipeline.generate_response(query, results)
        
        print(f"Summary: {response['summary']}")
        for i, quote in enumerate(response['retrieved_quotes'][:2], 1):
            print(f"  {i}. \"{quote['quote'][:100]}...\" - {quote['author']} (Score: {quote['similarity_score']:.3f})")
    
    print("\n=== Implementation Complete ===")
    print("To run the Streamlit app, use: streamlit run this_script.py")

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If we can import streamlit and we're in streamlit context
        create_streamlit_app()
    except:
        # If not in streamlit, run the main function
        main()