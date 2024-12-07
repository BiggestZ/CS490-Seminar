# Import functions from Base_Rag_App and LLM_Query
from Base_Rag_App import *
from LLM_Query import *
from pinecone import Pinecone

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

def extract_scores(query, top_k=5):
    """
    Extracts the top-k scores for both preprocessed and non-preprocessed query embeddings.

    :param query: The query string to compare.
    :param top_k: The number of top results to retrieve for comparison (default is 5).
    :return: A dictionary containing comparison metrics.
    """
    # Preprocess the query and embed both versions
    processed_query = preprocess_query(query)
    processed_query_embedding = get_query_embedding(processed_query)
    unprocessed_query_embedding = get_query_embedding_base(query)

    # Retrieve top-k results for processed query using hybrid retrieval
    processed_results = hybrid_retrieve_from_pinecone(processed_query_embedding, top_k=top_k)
    processed_scores = [(res["metadata"].get("title", "No Title"), res["score"]) for res in processed_results]

    # Retrieve top-k results for regular query directly from Pinecone using base embedding
    regular_results = index.query(vector=unprocessed_query_embedding, top_k=top_k, include_metadata=True)["matches"]
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

# Example usage
query = "What are some internship opportunities at Occidental College?"
comparison_results = extract_scores(query)

# Display comparison results
print("Comparison Results:", comparison_results)