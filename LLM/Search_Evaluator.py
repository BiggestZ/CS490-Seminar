# This file will provide the scores for processed and non-processed queries.

# Import the necessary libraries
import os
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from LLM_Query import * # The advanced RAG Search
from Base_Rag_App import * # The most base version

# Define constants (Pinecone API key, index name, and environment)
PINECONE_API_KEY = "b1a441c1-f979-4c43-86a3-9b9aac46aea2"
PINECONE_INDEX = "chunked-text-embeddings-3"
PINECONE_ENV = "us-east-1"

def connect_to_pinecone(api_key, index_name):
    """Connects to Pinecone and returns the index if available."""
    pc = Pinecone(api_key=api_key, environment=PINECONE_ENV)
    if index_name not in pc.list_indexes().names():
        raise ValueError("Index not found, please check spelling.")
    index = pc.Index(index_name)
    print("Connected to index")
    return index

def compare_query_results(query, preprocess_func, embedding_func, index, top_k=5):
    """
    Compare the scores of the top-k retrieved files for both preprocessed and regular queries.

    :param query: The original query string.
    :param preprocess_func: Function to preprocess the query.
    :param embedding_func: Function to embed the query.
    :param index: Pinecone index to retrieve results from.
    :param top_k: Number of top results to retrieve for comparison (default is 5).
    :return: Dictionary with comparison metrics.
    """
    # Preprocess the query and embed both versions
    processed_query = preprocess_func(query)
    processed_embedding = embedding_func(processed_query)
    regular_embedding = embedding_func(query)

    # Retrieve top-k results for processed query
    processed_results = index.query(vector=processed_embedding, top_k=top_k, include_metadata=True)["matches"]
    processed_scores = [(res["metadata"].get("title", "No Title"), res["score"]) for res in processed_results]

    # Retrieve top-k results for regular query
    regular_results = index.query(vector=regular_embedding, top_k=top_k, include_metadata=True)["matches"]
    regular_scores = [(res["metadata"].get("title", "No Title"), res["score"]) for res in regular_results]

    # Calculate average score difference
    avg_processed_score = sum(score for _, score in processed_scores) / top_k
    avg_regular_score = sum(score for _, score in regular_scores) / top_k
    avg_score_difference = avg_processed_score - avg_regular_score

    # Calculate overlap in top results
    processed_titles = {title for title, _ in processed_scores}
    regular_titles = {title for title, _ in regular_scores}
    overlap_count = len(processed_titles.intersection(regular_titles))
    overlap_percentage = (overlap_count / top_k) * 100

    # Compile comparison results
    comparison = {
        "Processed Scores": processed_scores,
        "Regular Scores": regular_scores,
        "Average Processed Score": avg_processed_score,
        "Average Regular Score": avg_regular_score,
        "Average Score Difference": avg_score_difference,
        "Top-K Overlap Count": overlap_count,
        "Top-K Overlap Percentage": overlap_percentage,
    }

    return comparison

def main():
    # Connect to Pinecone Index
    index = connect_to_pinecone(PINECONE_API_KEY, PINECONE_INDEX)
    
    # Load the embedding model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Check and download nltk libraries if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    # Define query and process
    query = "What are some internship opportunities at Occidental College?"
    
    # Get embeddings for processed and unprocessed queries
    processed_query = preprocess_query(query)  # LLM processed embedding
    unprocessed_query = query  # Regular embedding
    
    comparison_results = compare_query_results(unprocessed_query, processed_query, get_query_embedding, index)

    '''# Get embeddings for processed and unprocessed queries
    processed_query = preprocess_query(query)  # LLM processed embedding
    unprocessed_query = query  # Regular embedding

    print("Unprocessed Query Embedding:", unprocessed_query)
    print("Processed Query Embedding:", processed_query)
    
    # Retrieve and print results for processed query & Unprocessed query
    print("These are the results for the Processed Query:")
    print_results_llm(processed_query)

    print("These are the results for the Unprocessed Query:")
    print_query(unprocessed_query)
'''
# Run the main function only if this script is executed directly
if __name__ == "__main__":
    main()
