import os
from dotenv import load_dotenv
import pinecone
import random
import time
import uuid

# Load environment variables
load_dotenv()

def initialize_pinecone():
    """Initialize Pinecone connection."""
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    
    # Initialize Pinecone with new API
    pinecone_client = pinecone.Pinecone(api_key=api_key)
    
    # Check if index exists, if not create it
    index_name = "searchableapi"
    
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

def generate_random_embedding():
    """Generate a random embedding vector for testing."""
    return [random.random() for _ in range(1536)]

def store_test_vectors(index, num_vectors=5):
    """
    Store test vectors in Pinecone for testing.
    
    Args:
        index: Pinecone index
        num_vectors: Number of test vectors to store
    
    Returns:
        list: List of vector IDs stored
    """
    print(f"Storing {num_vectors} test vectors in Pinecone...")
    
    # Prepare vectors for upsert
    vectors = []
    vector_ids = []
    
    for i in range(num_vectors):
        # Generate random embedding
        embedding = generate_random_embedding()
        
        # Create metadata
        metadata = {
            "url": f"https://example.com/test/{i}",
            "title": f"Test Article {i}",
            "source_type": "Test",
            "publication_date": "2025-03-05",
            "authors": "Test Author",
            "journal": "Test Journal",
            "chunk_index": i,
            "total_chunks": num_vectors,
            "chunk_text": f"This is test chunk {i} for testing Pinecone vector storage."
        }
        
        # Create vector ID
        vector_id = f"test-{uuid.uuid4()}"
        vector_ids.append(vector_id)
        
        # Add to vectors list
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })
    
    # Upsert vectors to Pinecone
    try:
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
            print(f"Stored {min(i+batch_size, len(vectors))}/{len(vectors)} vectors")
        
        print(f"Successfully stored {len(vectors)} test vectors in Pinecone")
        
        # Add a delay to ensure vectors are indexed
        print("Waiting for vectors to be indexed...")
        time.sleep(5)
        
        return vector_ids
    except Exception as e:
        print(f"Error upserting vectors to Pinecone: {str(e)}")
        return []

def query_test_vectors(index, vector_ids):
    """
    Query test vectors from Pinecone.
    
    Args:
        index: Pinecone index
        vector_ids: List of vector IDs to query
    """
    print(f"Querying {len(vector_ids)} test vectors from Pinecone...")
    
    try:
        # Generate a random query vector
        query_vector = generate_random_embedding()
        
        # Query Pinecone
        results = index.query(
            vector=query_vector,
            top_k=10,
            include_metadata=True
        )
        
        print(f"Query results: {results}")
        
        # Fetch vectors by ID
        print("Fetching vectors by ID...")
        fetched_vectors = index.fetch(ids=vector_ids)
        
        print(f"Fetched {len(fetched_vectors.vectors)} vectors by ID")
        for id, vector in fetched_vectors.vectors.items():
            print(f"Vector ID: {id}")
            print(f"Metadata: {vector.metadata}")
            print("---")
        
        return True
    except Exception as e:
        print(f"Error querying Pinecone: {str(e)}")
        return False

def delete_test_vectors(index, vector_ids):
    """
    Delete test vectors from Pinecone.
    
    Args:
        index: Pinecone index
        vector_ids: List of vector IDs to delete
    """
    print(f"Deleting {len(vector_ids)} test vectors from Pinecone...")
    
    try:
        # Delete vectors
        index.delete(ids=vector_ids)
        print(f"Successfully deleted {len(vector_ids)} test vectors from Pinecone")
        return True
    except Exception as e:
        print(f"Error deleting vectors from Pinecone: {str(e)}")
        return False

def main():
    try:
        # Initialize Pinecone
        print("Initializing Pinecone...")
        index = initialize_pinecone()
        
        # Store test vectors
        vector_ids = store_test_vectors(index, num_vectors=5)
        
        if vector_ids:
            # Query test vectors
            query_test_vectors(index, vector_ids)
            
            # Delete test vectors
            delete_test_vectors(index, vector_ids)
        
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    main()
