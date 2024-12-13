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

# Member Variables (Ensure keys are correctly set):
if 1:  # Replace with valid keys
    OPENAI = ''
    PINECONE = ''  # Pinecone API key
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
client = OpenAI(api_key=OPENAI)

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
    tokens = nltk.word_tokenize(query)

    # Remove stopwords
    filtered_words = [
        word.lower()
        for word, pos in pos_tag(tokens)
        if word.lower() not in stop_words and
        (word[0].isupper() and pos == 'NNP' or pos.startswith('NN') or pos == 'JJ')
    ]
    print(f'This is the filtered words: {filtered_words} + X')
    return ' '.join(filtered_words)

# Import title calculator
def calculate_title_score(query, title):
    query = preprocess_query(query)
    query_words = set(query.lower().split())
    title_words = set(re.split('[-_.]', title.lower()))

    # Calculate the proportion of query words found in the title
    matching_words = query_words.intersection(title_words)
    print(f'This is the matching words: {matching_words} + X')
    return len(matching_words) / len(query_words)

# Import get_query_embedding
def get_query_embedding(query):
    processed_query = preprocess_query(query)
    query_embedding = model.encode(processed_query).tolist()
    return query_embedding

# Use the hybrid docs retrieval model to get the top 3 results
def hybrid_retrieve_from_pinecone_docs(query, top_k=5, title_boost_factor=1.5):
    query_embedding = get_query_embedding(query)

    # Perform semantic search with Pinecone
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    combined_results = []

    for match in response['matches']:
        text = match['metadata']['page_content']
        title = match['metadata'].get('title', "")

        # Combine embedding score and title-based score
        embedding_score = match['score']
        title_score = calculate_title_score(query, title)
        boosted_score = embedding_score + (title_boost_factor * title_score)

        combined_results.append({
            'text': text,
            'title': title,
            'boosted_score': boosted_score
        })

    combined_results = sorted(combined_results, key=lambda x: x['boosted_score'], reverse=True)

    # Extract the texts of the top-k results after boosting
    top_documents = [result['text'] for result in combined_results[:top_k]]
    return top_documents

# Calculate dynamic max tokens
def calc_max_tokens(context, base_tokens=500, max_length=2000):
    context_tokens = len(context.split())
    if context_tokens < 50:
        return base_tokens - 100
    elif context_tokens > 500:
        return min(base_tokens + (context_tokens // 2), max_length)
    else:
        return base_tokens

# Generate an answer with GPT-4o
def generate_answer_with_gpt4o(query):
    docs = hybrid_retrieve_from_pinecone_docs(query)

    context = "\n\n".join(docs)
    print("This is the context: ", context)

    system_prompt = (
        "You are a highly knowledgeable assistant with access to a range of documents about Occidental College. "
        "Use the following documents to answer the question accurately:\n\n"
        f"{context}\n\n"
        "Answer the following question based on the context provided. Please make it around 500 words long."
    )

    # Call GPT-4o API
    response = client.chat.completions.create(
        model="gpt-4o",  # Using GPT-4o model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=calc_max_tokens(context),
        temperature=0.2
    )

    try:
        answer = response.choices[0].message.content
        print(answer)
        return answer
    except (AttributeError, IndexError) as e:
        print("Error processing the response:", e)
        return None

# Generate an answer using GPT-4o
generate_answer_with_gpt4o("What is the process for reserving a study room in the library master calendar?")