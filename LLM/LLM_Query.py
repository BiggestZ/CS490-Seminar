from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import openai
import re

# Member Variables:
# REMEMBER TO REMOVE BEFORE PUSHING
if 1:
    OPENAI = ''
    PINECONE = ""  
    PINECONE_INDEX = "chunked-text-embeddings-3"  
    PINECONE_ENV = "us-east-1"  

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE)

# Initialize OpenAI
openai.api_key = OPENAI

# Check if the index exists
if PINECONE_INDEX not in pc.list_indexes().names():
    print('This index does not exist: Please check spelling')

# Connect to index
index = pc.Index(PINECONE_INDEX)

# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Check and download nltk libraries if not already downloaded
def nltkCheck():
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
# Uncomment when wanting to run the function
#  nltkCheck()

# Preprocess the query
def preprocess_query(query):
    stop_words = set(nltk.corpus.stopwords.words('english')) 
    #print(f'This is the Stop Words: {stop_words} + X')
    tokens = nltk.word_tokenize(query) 
    # print(f'This is the tokens: {tokens} + XZ')
    # print(f'This is the pos_tag: {pos_tag(tokens)} + XZ')

    # Remove stopwords
    filtered_words = []

    for word, pos in pos_tag(tokens):
        word_lower = word.lower()
        #print(f'This is the word: {word}')
        #print(f'This is the word[0]: {word[0].isupper()} + X')
        if word_lower not in stop_words and word[0].isupper() and pos == 'NNP' or pos.startswith('NN') or pos == 'JJ':
            filtered_words.append(word_lower)
    print(f'This is the filtered words: {filtered_words} + X')
    return ' '.join(filtered_words)

# Function to embed query using OpenAI (or any other embedding model you are using)
def get_query_embedding(query):
    # Preprocess the query
    processed_query = preprocess_query(query)
    print(f'This is the processed query: {processed_query} + X')
    # Embed query
    query_embedding = model.encode(processed_query).tolist()
    return query_embedding

# Function to boost the score based on title relevance
def calculate_title_score(query, title):
    # Simple approach: check if query words are in the title
    # Note: "president" != "presidents" -> Solution: Stemming / Lemmanization ( Title only, not for vector search )
    query = preprocess_query(query) # Preprocess query for title matching [NOTE: THIS CHANGE INCREASED SCORE, REMEMBER TO NOTE THIS IN PAPER]
    query_words = set(query.lower().split())
    title_words = set(re.split('[-_.]', title.lower())) # Split by hyphen, underscore, or period
    print(query_words)
    print(title_words)
    
    # Calculate the proportion of query words found in the title
    matching_words = query_words.intersection(title_words)
    print(f'This is the matching words: {matching_words} + X')
    
    # Boost score by matching word count or other criteria (e.g., Levenshtein distance for fuzzy matching)
    return len(matching_words) / len(query_words)  # Adjust this as needed for your boosting

# Function to retrieve and rank results based on both embedding and title matching
def hybrid_retrieve_from_pinecone_score(query, top_k=5, title_boost_factor=1.5):
    # Step 1: Get the query embedding
    # print(f'This is the query embedding(Numerical): {query} + XT')
    query_embedding = get_query_embedding(query)
    
    # Step 2: Perform semantic search with Pinecone using the query embedding
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True  # Assuming metadata contains both 'text' and 'title'
    )
    
    # Step 3: Combine scores from embedding similarity and title relevance
    combined_results = []
    
    for match in response['matches']:
        text = match['metadata']['page_content']
        #print(f'This is the page content: {text}') 
        title = match['metadata'].get('title', "")  # Default to empty if no title
        
        # Get the original score (similarity score from Pinecone)
        embedding_score = match['score']
        
        # Calculate a title-based score boost
        title_score = calculate_title_score(query, title)
        print(f'This is the title score: {title_score} + X')
        
        # Combine the embedding score with the title score boost
        boosted_score = embedding_score + (title_boost_factor * title_score)
        print(f'This is the boosted score: {boosted_score} + X')
        
        # Store the result with the boosted score
        combined_results.append({
            'text': text,
            'title': title,
            'boosted_score': boosted_score
        })
    
    # Step 4: Sort by the boosted score
    combined_results = sorted(combined_results, key=lambda x: x['boosted_score'], reverse=True)
    
    # Return the top-k results after boosting
    return combined_results

# By GPT for the Demo 
def hybrid_retrieve_from_pinecone_docs(query, top_k=5, title_boost_factor=1.5):
    # Step 1: Get the query embedding
    query_embedding = get_query_embedding(query)
    
    # Step 2: Perform semantic search with Pinecone using the query embedding
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True  # Assuming metadata contains both 'page_content' and 'title'
    )
    
    # Step 3: Combine scores from embedding similarity and title relevance
    combined_results = []
    
    for match in response['matches']:
        text = match['metadata']['page_content']
        title = match['metadata'].get('title', "")  # Default to empty if no title
        
        # Get the original score (similarity score from Pinecone)
        embedding_score = match['score']
        
        # Calculate a title-based score boost
        title_score = calculate_title_score(query, title)
        
        # Combine the embedding score with the title score boost
        boosted_score = embedding_score + (title_boost_factor * title_score)
        
        # Store the result with the boosted score
        combined_results.append({
            'text': text,
            'title': title,
            'boosted_score': boosted_score
        })
    # Step 4: Sort by the boosted score
    combined_results = sorted(combined_results, key=lambda x: x['boosted_score'], reverse=True)
    
    # Step 5: Extract the texts of the top-k results after boosting
    top_documents = [result['text'] for result in combined_results[:top_k]]
    
    # Return the list of document texts for use with ChatGPT
    return top_documents

# Example usage:
def print_results_llm(query):
    hybrid_results = hybrid_retrieve_from_pinecone_score(query)
    for result in hybrid_results:
        print(f"Title: {result['title']}, Boosted Score: {result['boosted_score']}\nText: {result['text']}\n")

# hybrid_retrieve_from_pinecone_score("Who is Justin Li?")
# Uncomment when wanting to run the function
print_results_llm("What are internship opportunities at Occidental College?")
# NOTE: ELIMINATE QUESTION MARKS FROM QUERYS