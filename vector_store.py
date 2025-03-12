import os
from dotenv import load_dotenv
import pinecone
from openai import OpenAI
import tiktoken
import time
import json
import uuid

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def initialize_pinecone():
    """Initialize Pinecone connection."""
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    
    # Initialize Pinecone with new API
    pinecone_client = pinecone.Pinecone(api_key=api_key)
    
    # Check if index exists, if not create it
    #index_name = "searchableapi"
    index_name = "demo"
    
    # Get list of indexes
    indexes = [index.name for index in pinecone_client.list_indexes()]
    
    if index_name not in indexes:
        # Create index
        pinecone_client.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine"
        )
        print(f"Created new Pinecone index: {index_name}")
    
    # Connect to index
    index = pinecone_client.Index(index_name)
    return index

def num_tokens(text):
    """Count the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
    return len(encoding.encode(text))

def chunk_text(text, chunk_size=512, chunk_overlap=64):
    """Split text into chunks with specified size and overlap."""
    if not text:
        return []
    
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        # Get chunk of tokens
        chunk_end = min(i + chunk_size, len(tokens))
        chunk = tokens[i:chunk_end]
        
        # Decode chunk back to text
        chunk_text = encoding.decode(chunk)
        chunks.append(chunk_text)
        
        # Move to next chunk, considering overlap
        i += (chunk_size - chunk_overlap)
    
    return chunks

def generate_embedding(text):
    """Generate embedding for a text using OpenAI API."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error generating embedding, retrying in {retry_delay}s: {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to generate embedding after {max_retries} attempts: {str(e)}")
                raise

def store_article_chunks(article, index, status_callback=None):
    """
    Process an article, chunk its content, generate embeddings, and store in Pinecone.
    
    Args:
        article (dict): Article data including URL, Title, Summary, etc.
        index: Pinecone index
        status_callback (function, optional): Callback function for status updates
    
    Returns:
        int: Number of chunks stored
    """
    try:
        # Extract content from article - use full Content instead of Summary
        content = article.get('Content', '')
        
        # If Content is not available, fall back to Summary
        if not content:
            content = article.get('Summary', '')
        
        # Log article details for debugging
        print(f"Processing article: {article.get('Title', 'Unknown')}")
        print(f"URL: {article.get('URL', 'No URL')}")
        print(f"Content length: {len(content) if content else 0} characters")
        
        # If content is too short, skip
        if not content or len(content) < 100:
            print(f"Content too short for article: {article.get('Title', 'Unknown')}")
            if status_callback:
                status_callback(f"Skipping article (content too short): {article.get('Title', 'Unknown')}", 100)
            return 0
        
        # Chunk the content
        chunks = chunk_text(content)
        print(f"Created {len(chunks)} chunks from article")
        
        if status_callback:
            status_callback(f"Chunking article: {article.get('Title', 'Unknown')} ({len(chunks)} chunks)", 0)
        
        # Prepare vectors for upsert
        vectors = []
        
        for i, chunk in enumerate(chunks):
            if status_callback:
                status_callback(f"Processing chunk {i+1}/{len(chunks)}", (i+1)/len(chunks)*100)
            
            # Generate embedding
            try:
                embedding = generate_embedding(chunk)
                
                # Create metadata
                metadata = {
                    "url": article.get('URL', ''),
                    "title": article.get('Title', ''),
                    "source_type": article.get('Source_Type', ''),
                    "publication_date": str(article.get('Publication_Date', '')),
                    "authors": article.get('Authors', ''),
                    "journal": article.get('Journal', ''),
                    "doi": article.get('DOI', ''),
                    "abstract": article.get('Abstract', ''),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_text": chunk
                }
                
                # Create vector ID
                vector_id = f"{uuid.uuid4()}"
                
                # Add to vectors list
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                if status_callback:
                    status_callback(f"Error processing chunk {i}: {str(e)}", (i+1)/len(chunks)*100)
    except Exception as e:
        print(f"Error in store_article_chunks: {str(e)}")
        if status_callback:
            status_callback(f"Error processing article: {str(e)}", 100)
        return 0
    
    # Upsert vectors to Pinecone
    if vectors:
        try:
            print(f"Upserting {len(vectors)} vectors to Pinecone")
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                try:
                    index.upsert(vectors=batch)
                    print(f"Successfully upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
                    
                    if status_callback:
                        status_callback(f"Stored {min(i+batch_size, len(vectors))}/{len(vectors)} vectors", 100)
                except Exception as e:
                    print(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
                    if status_callback:
                        status_callback(f"Error upserting batch: {str(e)}", 100)
                    # Continue with next batch instead of failing completely
            
            # Add a small delay to ensure vectors are indexed
            time.sleep(2)
            
            return len(vectors)
        except Exception as e:
            print(f"Error upserting vectors to Pinecone: {str(e)}")
            if status_callback:
                status_callback(f"Error upserting vectors: {str(e)}", 100)
            return 0
    
    return 0

def query_similar_chunks(query_text, index, top_k=5):
    """
    Query Pinecone for chunks similar to the query text.
    
    Args:
        query_text (str): Query text
        index: Pinecone index
        top_k (int): Number of results to return
    
    Returns:
        list: List of similar chunks with metadata
    """
    try:
        # Generate embedding for query
        query_embedding = generate_embedding(query_text)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract results
        matches = []
        for match in results.matches:
            matches.append({
                "score": match.score,
                "metadata": match.metadata
            })
        
        return matches
    except Exception as e:
        print(f"Error querying Pinecone: {str(e)}")
        return []

def get_relevant_context(query, index, max_tokens=1500):
    """
    Get relevant context for a query from Pinecone.
    
    Args:
        query (str): Query text
        index: Pinecone index
        max_tokens (int): Maximum number of tokens to include in context
    
    Returns:
        str: Relevant context
    """
    # Query Pinecone for similar chunks
    matches = query_similar_chunks(query, index, top_k=10)
    
    if not matches:
        return ""
    
    # Sort by score (highest first)
    matches.sort(key=lambda x: x["score"], reverse=True)
    
    # Build context
    context = []
    token_count = 0
    
    for match in matches:
        chunk_text = match["metadata"].get("chunk_text", "")
        chunk_tokens = num_tokens(chunk_text)
        
        # Check if adding this chunk would exceed max_tokens
        if token_count + chunk_tokens > max_tokens:
            break
        
        # Add enhanced source information with full metadata
        title = match['metadata'].get('title', 'Unknown')
        url = match['metadata'].get('url', 'No URL')
        authors = match['metadata'].get('authors', 'Unknown')
        abstract = match['metadata'].get('abstract', '')
        
        source_info = f"SOURCE_METADATA:\nTitle: {title}\nURL: {url}\nAuthors: {authors}"
        
        # Add abstract if available
        if abstract:
            source_info += f"\nAbstract: {abstract}"
            
        source_info += f"\n\nCONTENT:\n{chunk_text}"
        
        # Add to context
        context.append(source_info)
        token_count += chunk_tokens
    
    return "\n\n".join(context)
