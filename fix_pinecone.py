import os
from dotenv import load_dotenv
import pinecone
import json
import time
import traceback

# Load environment variables
load_dotenv()

def check_pinecone_quota():
    """Check Pinecone quota and usage."""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            print("Error: PINECONE_API_KEY environment variable is not set")
            return False
        
        # Initialize Pinecone client
        print("Initializing Pinecone client...")
        pinecone_client = pinecone.Pinecone(api_key=api_key)
        
        # List indexes
        print("Listing indexes...")
        indexes = [index.name for index in pinecone_client.list_indexes()]
        print(f"Available indexes: {indexes}")
        
        # Check if our index exists
        index_name = "searchableapi"
        if index_name not in indexes:
            print(f"Index '{index_name}' not found. Creating it...")
            pinecone_client.create_index(
                name=index_name,
                dimension=1536,
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
        stats = index.describe_index_stats()
        print(f"Index stats: {json.dumps(stats.to_dict(), indent=2)}")
        
        # Check if we're on the free tier (gcp-starter)
        environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
        if environment == "gcp-starter":
            print("\nYou are using the Pinecone free tier (gcp-starter)")
            print("Free tier limitations:")
            print("- 1 index")
            print("- 100,000 vectors maximum")
            print("- 10,000 operations per day")
            print("- 5 QPS (queries per second)")
            
            # Check if we're approaching limits
            vector_count = stats.total_vector_count
            print(f"\nCurrent vector count: {vector_count}/100,000")
            
            if vector_count >= 90000:
                print("WARNING: You are approaching the vector limit for the free tier!")
            
            print("\nPossible issues:")
            print("1. You may have exceeded your daily operation quota (10,000 operations)")
            print("2. You may be exceeding the QPS limit (5 queries per second)")
            print("3. There might be an issue with the vector format or metadata")
            
        return True
    except Exception as e:
        print(f"Error checking Pinecone quota: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def fix_vector_store_module():
    """Fix the vector_store.py module to properly handle Pinecone operations."""
    try:
        print("\nChecking vector_store.py for issues...")
        
        # Read the current file
        with open('vector_store.py', 'r') as f:
            content = f.read()
        
        # Check if we need to make changes
        changes_needed = False
        
        # Check for common issues
        if "pinecone.init" in content:
            print("Found outdated pinecone.init() call - needs to be updated to new Pinecone client")
            changes_needed = True
        
        if "pinecone.Index" in content and "pinecone_client.Index" not in content:
            print("Found outdated pinecone.Index() call - needs to be updated to pinecone_client.Index()")
            changes_needed = True
        
        if changes_needed:
            print("Creating backup of original file...")
            with open('vector_store.py.bak', 'w') as f:
                f.write(content)
            print("Backup created as vector_store.py.bak")
            
            print("Updating vector_store.py with fixes...")
            
            # Replace outdated initialization
            content = content.replace(
                "def initialize_pinecone():",
                """def initialize_pinecone():
    \"\"\"Initialize Pinecone connection.\"\"\"
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    
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
    return index"""
            )
            
            # Remove any old pinecone.init calls
            if "pinecone.init" in content:
                lines = content.split('\n')
                new_lines = []
                skip_block = False
                
                for line in lines:
                    if "pinecone.init" in line:
                        skip_block = True
                        continue
                    
                    if skip_block and (")" in line or line.strip() == ""):
                        skip_block = False
                        continue
                    
                    if not skip_block:
                        new_lines.append(line)
                
                content = '\n'.join(new_lines)
            
            # Write the updated file
            with open('vector_store.py', 'w') as f:
                f.write(content)
            
            print("vector_store.py has been updated with fixes")
        else:
            print("No issues found in vector_store.py")
        
        return True
    except Exception as e:
        print(f"Error fixing vector_store.py: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    print("=== PINECONE FIX SCRIPT ===")
    print("This script will diagnose and fix issues with Pinecone integration")
    
    # Check Pinecone quota
    print("\n1. Checking Pinecone quota and usage...")
    check_pinecone_quota()
    
    # Fix vector_store.py
    print("\n2. Checking and fixing vector_store.py...")
    fix_vector_store_module()
    
    print("\nFix completed. Please try running your application again.")
    print("If you're still experiencing issues, you may need to:")
    print("1. Check if you've exceeded your Pinecone free tier quota")
    print("2. Ensure your OpenAI API key is valid for generating embeddings")
    print("3. Check for rate limiting issues with the APIs")
    print("4. Verify that your data is properly formatted before storing")

if __name__ == "__main__":
    main()
