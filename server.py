from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import requests

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Initialize Qdrant client
# You can configure this with your Qdrant server details
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY", None)
)

# Collection name to use
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")

@app.route('/')
def index():
    """Serve the search interface as the index page."""
    return send_from_directory('.', 'search.html')

@app.route('/api/documents', methods=['GET'])
def get_all_documents():
    """
    Retrieve all documents from the Qdrant database.
    Supports pagination with limit and offset parameters.
    """
    try:
        # Get pagination parameters from query string
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        # Retrieve all points from the collection
        response = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            offset=offset,
            with_payload=True
        )
        
        # Format the response
        documents = []
        for point in response[0]:  # response[0] contains the points
            doc = {
                'id': point.id,
                'payload': point.payload,
                'vector': point.vector
            }
            documents.append(doc)
        
        return jsonify({
            'status': 'success',
            'data': documents,
            'next_page_offset': response[1]  # Offset for the next page
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """
    Search documents by text query and optional metadata filters.
    """
    try:
        data = request.json
        
        if not data or 'query' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Query text is required'
            }), 400
        
        query_text = data['query']
        limit = data.get('limit', 4)
        metadata_filters = data.get('filters', {})
        
        # Convert metadata filters to Qdrant filter format
        filter_conditions = []
        for key, value in metadata_filters.items():
            filter_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )
        
        # Create filter if there are any conditions
        search_filter = None
        if filter_conditions:
            search_filter = models.Filter(
                must=filter_conditions
            )
    
        query_vector = _get_embeddings(query_text)
        
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit
        )
        
        # Format the response
        results = []
        for hit in search_results:
            result = {
                'id': hit.id,
                'score': hit.score,
                'payload': hit.payload
            }
            results.append(result)
        
        return jsonify({
            'status': 'success',
            'data': results
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500



def _get_embeddings(text: str) -> List[float]:
    """Get embeddings from the OpenAI API.
        
    Args:
            text: Text to generate embeddings for
            
    Returns:
            List of embedding values
    """
    headers = {
        "Content-Type": "application/json"
    }
        
    data = {
        "input": text,
        "model": os.getenv("DEFAULT_EMBEDDING_MODEL")
    }
        
    response = requests.post(
        f"{os.getenv("API_BASE_URL")}/embeddings",
        headers=headers,
        json=data
    )
    response.raise_for_status()
        
    return response.json()["data"][0]["embedding"]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
