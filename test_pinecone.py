import os
from dotenv import load_dotenv
import pinecone
import random
import time

# Load environment variables
load_dotenv()

def test_pinecone_connection():
    """Test Pinecone connection and basic operations."""
    print("Testing Pinecone connection...")
    
    # Get API key from environment
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    
    if not api_key:
        print("Error: PINECONE_API_KEY environment variable is not set")
        return False
    
    try:
        # Initialize Pinecone with new API
        print("Initializing Pinecone client...")
        pinecone_client = pinecone.Pinecone(api_key=api_key)
        
        # List indexes
        print("Listing indexes...")
        indexes = [index.name for index in pinecone_client.list_indexes()]
        print(f"Available indexes: {indexes}")
        
        # Check if our index exists
        index_name = "searchableapi"
        if index_name not in indexes:
            print(f"Creating index '{index_name}'...")
            pinecone_client.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine"
            )
            print(f"Created new Pinecone index: {index_name}")
            # Wait for index to be ready
            time.sleep(10)
        
        # Connect to index
        print(f"Connecting to index '{index_name}'...")
        index = pinecone_client.Index(index_name)
        
        # Test upsert operation with a dummy vector
        print("Testing upsert operation...")
        test_id = f"test-vector-{int(time.time())}"
        dummy_vector = [random.random() for _ in range(1536)]  # Create a random vector with 1536 dimensions
        
        index.upsert(
            vectors=[
                {
                    "id": test_id,
                    "values": dummy_vector,
                    "metadata": {"test": "true"}
                }
            ]
        )
        print(f"Successfully upserted test vector with ID: {test_id}")
        
        # Test query operation
        print("Testing query operation...")
        results = index.query(
            vector=dummy_vector,
            top_k=1,
            include_metadata=True
        )
        
        print(f"Query results: {results}")
        
        # Delete test vector
        print("Cleaning up test vector...")
        index.delete(ids=[test_id])
        print("Test vector deleted")
        
        print("All Pinecone operations completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error testing Pinecone: {str(e)}")
        return False

if __name__ == "__main__":
    test_pinecone_connection()
