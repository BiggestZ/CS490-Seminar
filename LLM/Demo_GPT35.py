# This will be my Demo for Tuesday, 11/12
# import openai
from openai import OpenAI
# import LLM_Query
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import re
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Member Variables(Pls do not open):
if 1: # This is a different OPENAI KEY, MAKE SURE TO REMOVE
    OPENAI = ''
    PINECONE = ""  
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

# Initialize OpenAI Client
client = OpenAI(api_key = OPENAI)

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
nltkCheck()

# Import and preprocess the query
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
Processed_Query = preprocess_query("What are some internship opportunities at Occidental College?")

# Import title calculator
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

# Import get_query_embedding
def get_query_embedding(query):
    # Preprocess the query
    processed_query = preprocess_query(query)
    print(f'This is the processed query: {processed_query} + X')
    # Embed query
    query_embedding = model.encode(processed_query).tolist()
    return query_embedding

# Use the hybrid docs retrieval model to get the top 5 results
def hybrid_retrieve_from_pinecone_docs(query, top_k=3, title_boost_factor=1.5):
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

def calc_max_tokens(context, base_tokens = 500, max_length = 2000):
    context_tokens = len(context.split())
    print(f'This is the context tokens: {context_tokens} + X')
    if context_tokens < 50:
        return base_tokens - 100
    elif context_tokens > 500:
        return min(base_tokens + (context_tokens // 2), max_length)
    else:  # Medium-length context
        return base_tokens

def generate_answer_with_chatgpt(query):
    # Find Docs by performing hybrid search
    docs = hybrid_retrieve_from_pinecone_docs(query)

    # Combine the documents into a single context string
    context = "\n\n".join(docs)
    print("This is the context: ", context)

    # Prompt that includes context and query
    system_prompt = (
        "You are a highly knowledgeable assistant with access to a range of documents about Occidental College. "
        "Use the following documents to answer the question accurately:\n\n"
        f"{context}\n\n"
        "Answer the following question based on the context provided. Please make it around 500 words long."
    )

    # Call ChatGPT-4 API with context and user query
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=calc_max_tokens(context),
        temperature=0.2  # Lower temperature for factual responses
    )
    # Return the generated answer
    try:
        answer = response.choices[0].message.content
        print(answer)
        return answer
    except (AttributeError, IndexError) as e:
        print("Error processing the response:", e)
        return None

# Generate an answer using ChatGPT
generate_answer_with_chatgpt("What time does the gym close at Occidental College?")

