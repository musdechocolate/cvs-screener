import os
from typing import List, Dict, Any, Optional
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
import PyPDF2
import uuid
import time
import json


class MetaDataExtractor:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str
    ):
        """Initialize the MetaDataExtractor with API configurations.
        
        Args:
            base_url: Base URL for the OpenAI API endpoint
            api_key: API key for authentication
            model: The OpenAI model to use for extraction
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
    
    def _call_openai(self, text: str) -> Dict[str, Any]:
        """Call OpenAI API to extract metadata from text.
        
        Args:
            text: The CV text to analyze
            
        Returns:
            Dictionary containing extracted metadata
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/musdechocolate/HRAI", 
            "X-Title": "HRAI - CV Screener"
        }
        
        # Create a prompt that instructs the model to extract specific metadata
        prompt = """Extract the following information from the CV text in JSON format:
        - name: Full name of the candidate
        - age: Age if mentioned, otherwise null
        - years_of_experience: Total years of professional experience
        - skills: List of technical skills and tools
        - languages: List of programming languages
        - education: List of education details with degree, institution, and year
        - current_role: Current or most recent job title
        - location: Location if mentioned, otherwise null
        
        Return ONLY the JSON object, no additional text.
        If a field cannot be determined, use null.
        For lists, return empty arrays if no items found."""
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise CV parser that extracts structured metadata."},
                {"role": "user", "content": f"{prompt}\n\nCV Text:\n{text}"}
            ],
            "response_format": { "type": "json_object" }
        }
        
        try:
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=120  # Add timeout to prevent hanging
            )
            response.raise_for_status()
            
            # Parse the JSON response
            try:
                return json.loads(response.json()["choices"][0]["message"]["content"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Response content: {response.text}")
                raise Exception(f"Failed to parse OpenAI response: {str(e)}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            raise Exception(f"API request failed: {str(e)}")
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from CV text.
        
        Args:
            text: The CV text to analyze
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            metadata = self._call_openai(text)
            
            # Ensure all expected fields are present
            expected_fields = {
                "name": None,
                "age": None,
                "years_of_experience": None,
                "skills": [],
                "languages": [],
                "education": [],
                "current_role": None,
                "location": None
            }
            
            # Update with extracted data, keeping defaults for missing fields
            metadata = {**expected_fields, **metadata}
            
            return metadata
            
        except Exception as e:
            raise Exception(f"Failed to extract metadata: {str(e)}")


class Indexer:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        embedding_model: str,
        qdrant_url: str,
        collection_name: str = "documents"
    ):
        """Initialize the Indexer with API configurations.
        
        Args:
            base_url: Base URL for the OpenAI API endpoint
            api_key: API key for authentication
            model: The model to use for extraction
            embedding_model: The model to use for embeddings
            qdrant_url: URL of the Qdrant instance
            collection_name: Name of the collection in Qdrant
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.metadata_extractor = MetaDataExtractor(
            base_url=base_url,
            api_key=api_key,
            model=model
        )
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create the collection in Qdrant if it doesn't exist."""
        collections = self.qdrant_client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        
        if not exists:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=int(os.getenv("EMBEDDING_DIMENSION")),
                    distance=models.Distance.COSINE
                )
            )
    
    def _get_embeddings(self, text: str) -> List[float]:
        """Get embeddings from the OpenAI API.
        
        Args:
            text: Text to generate embeddings for
            
        Returns:
            List of embedding values
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "input": text,
            "model": self.embedding_model
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        return response.json()["data"][0]["embedding"]
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF {file_path}: {str(e)}")
    
    def index_file(self, file_path: str, id_field: str = None) -> None:
        """Index a PDF file into Qdrant.
        
        Args:
            file_path: Path to the PDF file
            id_field: Optional field to use as the point ID (ignored for PDFs)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(file_path)
        
        # Extract metadata
        metadata = self.metadata_extractor.extract_metadata(text)
        
        # Get embeddings
        embedding = self._get_embeddings(text)
        
        # Generate a UUID for the point ID
        point_id = str(uuid.uuid4())
        
        # Store in Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        "filename": os.path.basename(file_path),
                        "filepath": file_path,
                        "metadata": metadata
                    }
                )
            ]
        )
    
    def index_directory(self, directory_path: str, id_field: str = None) -> None:
        """Index all PDF files in a directory.
        
        Args:
            directory_path: Path to the directory containing PDF files
            id_field: Optional field to use as the point ID (ignored for PDFs)
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        for filename in os.listdir(directory_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                print(f"Indexing {filename}...")
                self.index_file(file_path, id_field)
                print(f"Completed indexing {filename}")
                time.sleep(1)
