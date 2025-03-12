import os
from dotenv import load_dotenv
import pinecone
import random
import time
import json
import uuid
import traceback

# Load environment variables
load_dotenv()

def initialize_pinecone():
    """Initialize Pinecone connection with detailed logging."""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
        
        print(f"API Key: {'*' * (len(api_key) - 4) + api_key[-4:] if api_key else 'Not found'}")
        print(f"Environment: {environment}")
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        
        # Initialize Pinecone with new API
        print("Creating Pinecone client...")
        pinecone_client = pinecone.Pinecone(api_key=api_key)
        
        # Check if index exists, if not create it
        index_name = "searchableapi"
        
        # Get list of indexes
        print("Listing indexes...")
        indexes = [index.name for index in pinecone_client.list_indexes()]
        print(f"Available indexes: {indexes}")
        
        if index_name not in indexes:
            # Create index
            print(f"Creating new index '{index_name}'...")
            pinecone_client.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
            print(f"Created new Pinecone index: {index_name}")
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            time.sleep(10)
        
        # Connect to index
        print(f"Connecting to index '{index_name}'...")
        index = pinecone_client.Index(index_name)
        
        # Get index stats
        try:
            stats = index.describe_index_stats()
            print(f"Index stats: {json.dumps(stats.to_dict(), indent=2)}")
        except Exception as e:
            print(f"Error getting index stats: {str(e)}")
        
        return index
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def generate_random_embedding():
    """Generate a random embedding vector for testing."""
    return [random.random() for _ in range(1536)]

def store_test_vectors(index, num_vectors=2):
    """
    Store test vectors in Pinecone for testing with detailed logging.
    
    Args:
        index: Pinecone index
        num_vectors: Number of test vectors to store
    
    Returns:
        list: List of vector IDs stored
    """
    print(f"\nStoring {num_vectors} test vectors in Pinecone...")
    
    # Prepare vectors for upsert
    vectors = []
    vector_ids = []
    
    for i in range(num_vectors):
        # Generate random embedding
        embedding = generate_random_embedding()
        
        # Create metadata
        metadata = {
            "url": f"https://example.com/test/{i}",
            "title": f"Debug Test Article {i}",
            "source_type": "Test",
            "publication_date": "2025-03-05",
            "authors": "Test Author",
            "journal": "Test Journal",
            "chunk_index": i,
            "total_chunks": num_vectors,
            "chunk_text": f"This is debug test chunk {i} for testing Pinecone vector storage."
        }
        
        # Create vector ID
        vector_id = f"debug-test-{uuid.uuid4()}"
        vector_ids.append(vector_id)
        
        # Add to vectors list
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })
    
    # Upsert vectors to Pinecone
    try:
        print(f"Prepared {len(vectors)} vectors for upsert")
        print(f"First vector ID: {vectors[0]['id']}")
        print(f"First vector metadata: {json.dumps(vectors[0]['metadata'], indent=2)}")
        print(f"First vector has {len(vectors[0]['values'])} dimensions")
        
        # Upsert vectors
        print("Upserting vectors to Pinecone...")
        index.upsert(vectors=vectors)
        
        print(f"Successfully stored {len(vectors)} test vectors in Pinecone")
        
        # Add a delay to ensure vectors are indexed
        print("Waiting for vectors to be indexed...")
        time.sleep(5)
        
        # Get index stats after upsert
        try:
            stats = index.describe_index_stats()
            print(f"Index stats after upsert: {json.dumps(stats.to_dict(), indent=2)}")
        except Exception as e:
            print(f"Error getting index stats after upsert: {str(e)}")
        
        return vector_ids
    except Exception as e:
        print(f"Error upserting vectors to Pinecone: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return []

def query_test_vectors(index, vector_ids):
    """
    Query test vectors from Pinecone with detailed logging.
    
    Args:
        index: Pinecone index
        vector_ids: List of vector IDs to query
    """
    print(f"\nQuerying {len(vector_ids)} test vectors from Pinecone...")
    
    try:
        # Generate a random query vector
        query_vector = generate_random_embedding()
        
        # Query Pinecone
        print("Executing query...")
        results = index.query(
            vector=query_vector,
            top_k=10,
            include_metadata=True
        )
        
        print(f"Query results: {json.dumps(results.to_dict(), indent=2)}")
        
        # Fetch vectors by ID
        print(f"\nFetching vectors by ID: {vector_ids}")
        fetched_vectors = index.fetch(ids=vector_ids)
        
        print(f"Fetched {len(fetched_vectors.vectors)} vectors by ID")
        for id, vector in fetched_vectors.vectors.items():
            print(f"Vector ID: {id}")
            print(f"Metadata: {json.dumps(vector.metadata, indent=2)}")
            print("---")
        
        return True
    except Exception as e:
        print(f"Error querying Pinecone: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def delete_test_vectors(index, vector_ids):
    """
    Delete test vectors from Pinecone with detailed logging.
    
    Args:
        index: Pinecone index
        vector_ids: List of vector IDs to delete
    """
    print(f"\nDeleting {len(vector_ids)} test vectors from Pinecone...")
    
    try:
        # Delete vectors
        index.delete(ids=vector_ids)
        print(f"Successfully deleted {len(vector_ids)} test vectors from Pinecone")
        
        # Get index stats after delete
        try:
            stats = index.describe_index_stats()
            print(f"Index stats after delete: {json.dumps(stats.to_dict(), indent=2)}")
        except Exception as e:
            print(f"Error getting index stats after delete: {str(e)}")
        
        return True
    except Exception as e:
        print(f"Error deleting vectors from Pinecone: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    try:
        print("=== PINECONE DEBUG SCRIPT ===")
        print("This script will test Pinecone connection and operations with detailed logging")
        
        # Initialize Pinecone
        print("\n1. Initializing Pinecone...")
        index = initialize_pinecone()
        
        if not index:
            print("Failed to initialize Pinecone. Exiting.")
            return
        
        # Store test vectors
        vector_ids = store_test_vectors(index, num_vectors=2)
        
        if vector_ids:
            # Query test vectors
            query_test_vectors(index, vector_ids)
            
            # Delete test vectors
            delete_test_vectors(index, vector_ids)
        else:
            print("Failed to store test vectors. Exiting.")
            return
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error during test: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
