import pandas as pd
import os
import tiktoken
from typing import List, Dict, Optional
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import hashlib
import time
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'resume-embeddings')
EMBEDDING_MODEL = "text-embedding-3-small"  # Supports dimensions parameter, max 1536 dimensions
EMBEDDING_DIMENSIONS = 1024  # Match Pinecone index dimensions
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))  # tokens
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))  # tokens

# Rate limiting configuration
REQUESTS_PER_MINUTE = 50  # Conservative rate limit for OpenAI
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE  # Delay between requests

class ResumeEmbeddingPipeline:
    def __init__(self):
        # Initialize OpenAI client
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Pinecone
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Initialize tokenizer for chunking
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        self.index = None
        
    def create_or_connect_index(self, dimension: int = 1024):
        """Create or connect to Pinecone index with 1024 dimensions"""
        try:
            # Check if index exists
            if INDEX_NAME in self.pc.list_indexes().names():
                print(f"Connecting to existing index: {INDEX_NAME}")
                self.index = self.pc.Index(INDEX_NAME)
            else:
                print(f"Creating new index: {INDEX_NAME}")
                self.pc.create_index(
                    name=INDEX_NAME,
                    dimension=dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=PINECONE_ENVIRONMENT
                    )
                )
                # Wait for index to be ready
                while not self.pc.describe_index(INDEX_NAME).status['ready']:
                    time.sleep(1)
                self.index = self.pc.Index(INDEX_NAME)
                
            print(f"Index stats: {self.index.describe_index_stats()}")
            
        except Exception as e:
            print(f"Error creating/connecting to index: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing HTML tags and normalizing whitespace"""
        if pd.isna(text):
            return ""
        
        # If it looks like HTML, parse it
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def chunk_text(self, text: str, category: str) -> List[Dict]:
        """Chunk text into smaller pieces with overlap"""
        if not text.strip():
            return []
        
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            # Calculate end position
            end = min(start + CHUNK_SIZE, len(tokens))
            
            # Get chunk tokens and decode
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk metadata
            chunk_data = {
                'text': chunk_text,
                'chunk_id': chunk_id,
                'start_token': start,
                'end_token': end,
                'total_tokens': len(chunk_tokens),
                'category': category
            }
            
            chunks.append(chunk_data)
            
            # Move start position with overlap
            start = end - CHUNK_OVERLAP
            chunk_id += 1
            
            # Break if we've reached the end
            if end == len(tokens):
                break
        
        return chunks
    
    def get_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """Get embedding from OpenAI with rate limiting and retry logic"""
        for attempt in range(max_retries):
            try:
                # Rate limiting: wait before each request
                time.sleep(REQUEST_DELAY)
                
                response = self.openai_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text,
                    dimensions=EMBEDDING_DIMENSIONS  # Explicitly set dimensions to 1024
                )
                return response.data[0].embedding
                
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limiting (429 errors)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    wait_time = min(60 * (2 ** attempt), 300)  # Exponential backoff, max 5 minutes
                    print(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                
                # Handle other recoverable errors
                elif "timeout" in error_str.lower() or "connection" in error_str.lower():
                    wait_time = 5 * (attempt + 1)
                    print(f"Connection error, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                
                # For other errors, log and re-raise
                else:
                    print(f"Error getting embedding (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2)
        
        raise Exception(f"Failed to get embedding after {max_retries} attempts")
    
    def create_vector_id(self, category: str, resume_index: int, chunk_id: int) -> str:
        """Create a unique vector ID"""
        return f"{category}_{resume_index}_{chunk_id}"
    
    def process_resumes(self, csv_path: str, batch_size: int = 20, max_resumes: Optional[int] = None):
        """Process resumes from CSV and upsert to Pinecone with rate limiting"""
        try:
            # Load the dataset
            print(f"Loading dataset from {csv_path}...")
            df = pd.read_csv(csv_path)
            
            # Limit dataset for testing if specified
            if max_resumes:
                df = df.head(max_resumes)
                print(f"Limited to first {max_resumes} resumes for testing")
            
            print(f"Loaded {len(df)} resumes")
            
            # Create or connect to index
            self.create_or_connect_index()
            
            vectors_to_upsert = []
            total_chunks = 0
            processed_resumes = 0
            
            for idx, row in df.iterrows():
                try:
                    category = row['Category']
                    
                    # Use Resume_html if available, otherwise Resume_str
                    if 'Resume_html' in df.columns and pd.notna(row['Resume_html']):
                        resume_text = self.clean_text(row['Resume_html'])
                    elif 'Resume_str' in df.columns and pd.notna(row['Resume_str']):
                        resume_text = self.clean_text(row['Resume_str'])
                    else:
                        print(f"No resume text found for row {idx}")
                        continue
                    
                    if not resume_text.strip():
                        print(f"Empty resume text for row {idx}")
                        continue
                    
                    # Chunk the resume
                    chunks = self.chunk_text(resume_text, category)
                    
                    for chunk in chunks:
                        # Get embedding
                        embedding = self.get_embedding(chunk['text'])
                        
                        # Create vector ID
                        vector_id = self.create_vector_id(category, idx, chunk['chunk_id'])
                        
                        # Create metadata
                        metadata = {
                            'category': category,
                            'resume_index': idx,
                            'chunk_id': chunk['chunk_id'],
                            'text': chunk['text'][:1000],  # Truncate text for metadata
                            'total_tokens': chunk['total_tokens'],
                            'start_token': chunk['start_token'],
                            'end_token': chunk['end_token']
                        }
                        
                        # Add to batch
                        vectors_to_upsert.append({
                            'id': vector_id,
                            'values': embedding,
                            'metadata': metadata
                        })
                        
                        total_chunks += 1
                    
                    processed_resumes += 1
                    
                    # Upsert batch when it reaches batch_size
                    if len(vectors_to_upsert) >= batch_size:
                        print(f"Upserting batch of {len(vectors_to_upsert)} vectors...")
                        self.index.upsert(vectors_to_upsert)
                        vectors_to_upsert = []
                        time.sleep(0.1)  # Small delay to avoid rate limits
                    
                    if processed_resumes % 10 == 0:
                        print(f"Processed {processed_resumes} resumes, {total_chunks} chunks...")
                        
                except Exception as e:
                    print(f"Error processing resume {idx}: {e}")
                    continue
            
            # Upsert remaining vectors
            if vectors_to_upsert:
                print(f"Upserting final batch of {len(vectors_to_upsert)} vectors...")
                self.index.upsert(vectors_to_upsert)
            
            print(f"\nProcessing complete!")
            print(f"Processed {processed_resumes} resumes")
            print(f"Created {total_chunks} chunks")
            print(f"Index stats: {self.index.describe_index_stats()}")
            
        except Exception as e:
            print(f"Error in processing: {e}")
            raise

def main():
    """Main function to run the embedding pipeline"""
    # Check for required environment variables
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY environment variable is required")
        print("Set it with: export PINECONE_API_KEY='your-api-key'")
        return
    
    # Initialize pipeline
    pipeline = ResumeEmbeddingPipeline()
    
    # Process resumes
    csv_path = 'src/storage/resume/Resume.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    print("Starting resume embedding pipeline...")
    # Start with a small test batch to verify everything works
    pipeline.process_resumes(csv_path, batch_size=10, max_resumes=50)

if __name__ == "__main__":
    main()
