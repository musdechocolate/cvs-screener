import os
from dotenv import load_dotenv
from indexer import Indexer

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get configuration from environment variables
    base_url = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("API_KEY", "")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("COLLECTION_NAME", "documents")
    model = os.getenv("DEFAULT_LLM_MODEL", "qwen2.5-coder:14b")
    embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "snowflake-arctic-embed:latest")

    # Validate required environment variables
    if not base_url:
        raise ValueError(
            "Please set BASE_URL and API_KEY environment variables. "
            "You can create a .env file with these variables."
        )
    
    # Initialize the indexer
    indexer = Indexer(
        base_url=base_url,
        api_key=api_key,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        model=model,
        embedding_model=embedding_model
    )
    
    # Index all PDF files in the cvs directory
    try:
        indexer.index_directory("cvs", id_field=None)  # id_field is ignored for PDFs
        print("Successfully indexed all PDF files in the cvs directory")
    except FileNotFoundError:
        print("No PDF files found in the cvs directory")
    except Exception as e:
        print(f"An error occurred while indexing: {str(e)}")

if __name__ == "__main__":
    main() 