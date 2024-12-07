# This program serves as a basic rag, with no additional features.
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Member Variables:
if 1:
    PINECONE = "b1a441c1-f979-4c43-86a3-9b9aac46aea2"  
    PINECONE_INDEX = "chunked-text-embeddings-3"  
    PINECONE_ENV = "us-east-1"  

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE)

# Check if the index exists
if PINECONE_INDEX not in pc.list_indexes().names():
    print('This index does not exist: Please check spelling')

# Connect to index
index = pc.Index(PINECONE_INDEX)
print('Connected to index')

# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Function to embed query using OpenAI (or any other embedding model you are using)
def get_query_embedding_base(query):
    # Embed Query
    query_embedding = model.encode(query).tolist()
    return query_embedding

# Example usage:
query = "What are some internship opportunities at Occidental College?"
def print_query(query):
    qe = get_query_embedding_base(query)
    hresults = index.query(vector=qe, top_k=5, include_metadata=True)
    for result in hresults["matches"]:  # Access 'matches' in the results
        chunk_name = result["metadata"].get("title", "Unknown Title")  # Get the title from metadata
        score = result["score"]  # Get the score
        page_content = result["metadata"].get("page_content", "No content available")  # Get content if stored in metadata
        # Print details
        print(f"Chunk Name: {chunk_name}")
        print(f"Score: {score}")
        print(f"Text: {page_content}...")  # Print first 200 characters of content for brevity
        print("\n")

# Call the print_query function
# print_query(query)

