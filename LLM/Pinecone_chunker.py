import os
import pinecone
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Initialize Pinecone
api_key = 'b1a441c1-f979-4c43-86a3-9b9aac46aea2'
environment = 'us-east-1'
pinecone.init(api_key=api_key, environment=environment)

# Connect or create index in Pinecone
index_name = 'chunked-text-embeddings-4'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=768, metric='cosine')

index = pinecone.Index(index_name)

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Define text chunker
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# Function to process each text file and return LangChain Documents with chunks
def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into chunks
    chunks = text_splitter.split_text(content)
    title = os.path.basename(file_path)
    
    # Create LangChain documents with metadata for each chunk
    documents = [
        Document(page_content=chunk, metadata={"title": title, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    
    return documents

# Directory path containing text files
directory = 'Oxycrawler/web_data_2'

# Validate directory and text files
if not os.path.exists(directory):
    raise FileNotFoundError(f"The directory {directory} does not exist.")
text_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
if not text_files:
    raise FileNotFoundError(f"No text files found in the directory {directory}.")

# Process, embed, and upsert each file's chunks with metadata
for file_path in text_files:
    documents = process_text_file(file_path)  # Get LangChain documents with chunks and metadata
    
    for doc in documents:
        # Embed the chunk
        chunk_embedding = model.encode(doc.page_content).tolist()

        # Create a unique ID for each chunk
        chunk_id = f"{doc.metadata['title']}_chunk_{doc.metadata['chunk_index']}"

        # Upsert into Pinecone with metadata
        index.upsert([(chunk_id, chunk_embedding, doc.metadata)])

print("Chunks and embeddings with metadata have been stored in Pinecone.")


# OLD CODE: ABOVE IS NEW ONE
'''# This file serves to chunk text files and store the embeddings in  a Pinecone DB.
# File got lost in the rebase, will rewrite later.

import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader

# Step 1: Initialize Pinecone using the new Pinecone class
if 1:
    api_key = 'b1a441c1-f979-4c43-86a3-9b9aac46aea2'  # Replace with your actual Pinecone API key
    environment = 'us-east-1'  # Replace with your Pinecone environment or region

pc = Pinecone(api_key=api_key)

# Step 2: Create or connect to a Pinecone index
index_name = 'chunked-text-embeddings-3'
if index_name not in pc.list_indexes().names():
    # Create an index with 384 dimensions (for MiniLM-L6-v2)
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=environment)
    )

# Connect to the index
index = pc.Index(index_name)

# Step 3: Load the embedding model (Sentence-BERT)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') 

# Step 4: Define your text chunker (RecursiveCharacterTextSplitter) [May switch to nltk]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,    # Maximum characters per chunk
    chunk_overlap=100   # Overlap between chunks
)

# Function to read and chunk text from files
def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Extract the title of the text file
        title = os.path.basename(file_path)
        # Split the text into chunks
        chunks = text_splitter.split_text(content)
    return chunks

# Directory where your text files are stored (replace with your own directory)
directory = 'Oxycrawler/web_data_2'

# Check if the directory exists
if not os.path.exists(directory):
    raise FileNotFoundError(f"The directory {directory} does not exist.")

text_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]

# Check if there are any text files in the directory
if not text_files:
    raise FileNotFoundError(f"No text files found in the directory {directory}.")

print(len(text_files))

# Step 5: Chunk, embed, and store in Pinecone
for file_path in text_files:
    chunks = process_text_file(file_path)
    
    for i, chunk in enumerate(chunks):
        # Embed each chunk
        chunk_embedding = model.encode(chunk).tolist()

        # Create a unique ID for each chunk (file name + chunk index)
        chunk_id = f"{os.path.basename(file_path)}_chunk_{i}"

        # Upsert the chunk embedding into Pinecone
        index.upsert([(chunk_id, chunk_embedding)])

print("Chunks and embeddings have been stored in Pinecone.")

def text_to_document(text: str) -> Document:
    """
    Convert a text file into a LangChain Document with text and metadata.
    Args:
        file_path (str): Path to the text file.
    Returns:
        Document: A LangChain Document containing text and metadata.
    """
    # Extract the file name to use as the title or metadata
    title = os.path.basename(file_path)
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()

    # Create a LangChain Document with text and metadata
    doc = Document(
        page_content=content,  # The text content of the file
        metadata={"title": title}  # Metadata with file title
    )
    
    return doc

def upsert_chunk_with_metadata():
    for i, chunk in enumerate(chunks):
        # Embed each chunk
        chunk_embedding = model.encode(chunk).tolist()

        metadata = {
            "text": chunk,
            "title": "TOP" # Need to access the file name fromt the textfile, based on the chunks
        }
        # Create a unique ID for each chunk (file name + chunk index)
        chunk_id = f"{os.path.basename(file_path)}_chunk_{i}"

        # Upsert the chunk embedding into Pinecone
        index.upsert([(chunk_id, chunk_embedding)])
    pass
'''