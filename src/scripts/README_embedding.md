# Resume Embedding Pipeline

This script processes resumes from `Resume.csv`, chunks them, creates embeddings using OpenAI's `text-embedding-3-large` model, and stores them in Pinecone vector database.

## Setup

1. **Install dependencies**:
   ```bash
   pip install openai pinecone-client tiktoken beautifulsoup4 lxml
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export PINECONE_API_KEY="your-pinecone-api-key"
   export PINECONE_ENVIRONMENT="us-east-1"  # Optional, defaults to us-east-1
   ```

   Or create a `.env` file (copy from `.env.example` and fill in your keys).

3. **Ensure Resume.csv is in the correct location**:
   ```
   src/storage/resume/Resume.csv
   ```

## Usage

```bash
python src/scripts/embed_resumes_to_pinecone.py
```

## Configuration

You can modify these settings in the script:
- `INDEX_NAME`: Pinecone index name (default: "resume-embeddings")
- `EMBEDDING_MODEL`: OpenAI model (default: "text-embedding-3-large")
- `CHUNK_SIZE`: Token size per chunk (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## What the script does:

1. **Data Loading**: Loads resumes from `Resume.csv`
2. **Text Cleaning**: Removes HTML tags and normalizes whitespace
3. **Chunking**: Splits long resumes into smaller chunks with overlap
4. **Embedding**: Creates vector embeddings using OpenAI
5. **Storage**: Upserts embeddings to Pinecone with metadata

## Output

The script will create/update a Pinecone index with:
- Vector embeddings of resume chunks
- Metadata including category, chunk information, and text snippets
- Unique IDs for each chunk in format: `{category}_{resume_index}_{chunk_id}`

## Monitoring

The script provides progress updates including:
- Number of resumes processed
- Number of chunks created
- Batch upsert progress
- Final index statistics
