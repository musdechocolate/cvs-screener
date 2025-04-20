# CV screener

A Flask-based API server for interacting with a Qdrant vector database. This server provides endpoints for retrieving documents and searching by text query with optional metadata filters.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory with the following variables:
   ```
    API_BASE_URL=http://localhost:11434/v1
    DEFAULT_EMBEDDING_MODEL=snowflake-arctic-embed:latest
    DEFAULT_LLM_MODEL=
    API_KEY=

    QDRANT_URL=http://localhost:6333
    COLLECTION_NAME=documents
   ```

3. Run Qdrant instance and Ollama service with docker compose
   ```
   docker compose up -d
   ```

4. You can then index the CVs in the directory of CVs using the index.py
   ```
   python index.py
   ```

5. Make sure you have a running Qdrant instance with a collection named "documents" (or the name specified in your .env file).

## Running the Server

Start the Flask server with:
```
python server.py
```

The server will run on `http://localhost:5000` by default.

## API Endpoints

### 1. Get All Documents

Retrieves all documents from the Qdrant database with pagination support.

**Endpoint:** `GET /api/documents`

**Query Parameters:**
- `limit` (optional): Maximum number of documents to return (default: 100)
- `offset` (optional): Offset for pagination (default: 0)

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": "document_id",
      "payload": { ... },
      "vector": [ ... ]
    },
    ...
  ],
  "next_page_offset": "offset_for_next_page"
}
```

### 2. Search Documents

Searches for documents based on a text query and optional metadata filters.

**Endpoint:** `POST /api/search`

**Request Body:**
```json
{
  "query": "search text",
  "limit": 10,
  "filters": {
    "metadata_field": "value",
    "another_field": "another_value"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": "document_id",
      "score": 0.95,
      "payload": { ... }
    },
    ...
  ]
}
```
