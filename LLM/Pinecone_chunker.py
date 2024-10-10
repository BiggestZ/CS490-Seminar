import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader

# Step 1: Initialize Pinecone using the new Pinecone class
api_key = 'b1a441c1-f979-4c43-86a3-9b9aac46aea2'  # Replace with your actual Pinecone API key
environment = 'us-east-1'  # Replace with your Pinecone environment or region

pc = Pinecone(api_key=api_key)

# Step 2: Create or connect to a Pinecone index
index_name = 'chunked-text-embeddings-2.0'
if index_name not in pc.list_indexes().names():
    # Create an index with 384 dimensions (for MiniLM-L6-v2)
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=environment)
    )

# Connect to the index
index = pc.Index(index_name)

# Step 3: Load the embedding model (Sentence-BERT)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Step 4: Define your text chunker (RecursiveCharacterTextSplitter) [May switch to nltk]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # Maximum characters per chunk
    chunk_overlap=50   # Overlap between chunks
)

# Function to read and chunk text from files
def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Extract the title of the text file
        title = os.path.basename(file_path).replace('.txt', '').replace('campaign.oxy.edu-', ' ')
        # Split the text into chunks
        chunks = text_splitter.split_text(content)
    return chunks

# Directory where your text files are stored (replace with your own directory)
directory = 'Oxycrawler/web_data'

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

# Step 6: Query the stored embeddings with a new query
query = "How does Langchain help with large language models?"
query_embedding = model.encode(query).tolist()

# Search for the top 3 most similar chunks
result = index.query(queries=[query_embedding], top_k=3)

# Step 7: Display the results
print(f"Top {len(result['matches'])} most similar chunks:")
for match in result['matches']:
    print(f"Chunk ID: {match['id']} with score: {match['score']}")



# Code for later:

'''
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
'''
'''
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