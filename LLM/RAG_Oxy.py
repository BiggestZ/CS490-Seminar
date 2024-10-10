import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np

# Init Pinecone as a class
key = 'b1a441c1-f979-4c43-86a3-9b9aac46aea2'
pc = Pinecone(api_key=key)

# Connect to the index
index_name = 'chunked-text-embeddings'
if index_name not in pc.list_indexes().names():
    print('This index does not exist: Please check spelling')
    # Create an index with 384 dimensions (for MiniLM-L6-v2)
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
else:
    print('Index exists! Connecting...')
    index = pc.Index(index_name)
    index_description = pc.describe_index(index_name)
    index_dimension = index_description['dimension']
    print(f"The Dimensions of the index: {index_dimension}")

# Load the embedding model for the query
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Query the index
qu = 'Choi Auditorium'
query_embedding = model.encode(qu).tolist()
print(query_embedding)
#query_embedding = np.array(query_embedding).reshape(1, -1) # Reshape to 2D array

# Search the index
results = index.query(vector=query_embedding, top_k=5)
print(results)
