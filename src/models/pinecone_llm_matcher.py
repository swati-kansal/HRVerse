#!/usr/bin/env python3
"""
Pinecone + LLM Resume Matcher
Uses OpenAI embeddings and Pinecone vector database for semantic resume matching.
"""

import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'ai-hiring-portal')
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1024

# Global clients (initialized lazily)
_openai_client = None
_pinecone_index = None
_initialization_error = None


def _initialize_clients():
    """Lazily initialize OpenAI and Pinecone clients."""
    global _openai_client, _pinecone_index, _initialization_error
    
    if _initialization_error:
        return False
    
    if _openai_client is not None and _pinecone_index is not None:
        return True
    
    try:
        # Initialize OpenAI
        if not OPENAI_API_KEY:
            _initialization_error = "OPENAI_API_KEY not set"
            print(f"⚠️ Pinecone+LLM: {_initialization_error}")
            return False
        
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized")
        
        # Initialize Pinecone
        if not PINECONE_API_KEY:
            _initialization_error = "PINECONE_API_KEY not set"
            print(f"⚠️ Pinecone+LLM: {_initialization_error}")
            return False
        
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Connect to index
        if INDEX_NAME not in pc.list_indexes().names():
            _initialization_error = f"Pinecone index '{INDEX_NAME}' not found"
            print(f"⚠️ Pinecone+LLM: {_initialization_error}")
            return False
        
        _pinecone_index = pc.Index(INDEX_NAME)
        stats = _pinecone_index.describe_index_stats()
        print(f"✅ Connected to Pinecone index: {INDEX_NAME} ({stats['total_vector_count']} vectors)")
        
        return True
        
    except Exception as e:
        _initialization_error = str(e)
        print(f"⚠️ Pinecone+LLM initialization error: {e}")
        return False


def create_embedding(text: str) -> Optional[List[float]]:
    """
    Create embedding for text using OpenAI API.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding vector, or None on error
    """
    if not _initialize_clients():
        return None
    
    try:
        response = _openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSIONS
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Error creating embedding: {e}")
        return None


def search_similar_resumes(query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    """
    Search Pinecone for similar resumes.
    
    Args:
        query_embedding: The query embedding vector
        top_k: Number of results to return
        
    Returns:
        List of match dictionaries with scores and metadata
    """
    if not _initialize_clients() or _pinecone_index is None:
        return []
    
    try:
        query_response = _pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        matches = getattr(query_response, 'matches', [])
        results = []
        
        for match in matches:
            if isinstance(match, dict):
                score = match['score']
                metadata = match.get('metadata', {})
            else:
                score = match.score
                metadata = getattr(match, 'metadata', {})
            
            results.append({
                'score': score,
                'category': metadata.get('category', 'Unknown'),
                'resume_index': metadata.get('resume_index', 0),
                'text_preview': metadata.get('text', '')[:200]
            })
        
        return results
        
    except Exception as e:
        print(f"⚠️ Error searching Pinecone: {e}")
        return []


def calculate_match_score(cosine_similarity: float) -> Tuple[bool, int]:
    """
    Convert cosine similarity to match prediction and confidence.
    
    Args:
        cosine_similarity: Cosine similarity score (-1 to 1)
        
    Returns:
        Tuple of (is_match: bool, confidence: int)
    """
    # Convert cosine similarity to percentage (0-100)
    # Cosine similarity of 1.0 = 100% match
    # Cosine similarity of 0.0 = 50% match
    percentage = ((cosine_similarity + 1) / 2) * 100
    
    # Determine match threshold
    # Using 65% as threshold for "Match"
    is_match = percentage >= 65
    
    # Confidence is the percentage, capped at 98%
    confidence = min(int(percentage), 98)
    
    return is_match, confidence


def predict_match_pinecone_llm(resume_text: str, job_skills: str) -> Dict:
    """
    Predict resume-job match using Pinecone + LLM embeddings.
    
    This function:
    1. Creates embeddings for both resume and job description
    2. Calculates cosine similarity between the embeddings
    3. Returns match prediction with confidence
    
    Args:
        resume_text: The full resume text
        job_skills: Job skills/requirements as string
        
    Returns:
        Dictionary with model name, prediction, and confidence
    """
    # Check if clients are available
    if not _initialize_clients():
        return _fallback_prediction(resume_text, job_skills)
    
    try:
        # Create embeddings for both resume and job description
        resume_embedding = create_embedding(resume_text[:8000])  # Limit text length
        job_embedding = create_embedding(f"Required Skills: {job_skills}")
        
        if resume_embedding is None or job_embedding is None:
            return _fallback_prediction(resume_text, job_skills)
        
        # Calculate cosine similarity manually
        import numpy as np
        resume_vec = np.array(resume_embedding)
        job_vec = np.array(job_embedding)
        
        # Cosine similarity
        dot_product = np.dot(resume_vec, job_vec)
        norm_product = np.linalg.norm(resume_vec) * np.linalg.norm(job_vec)
        cosine_sim = dot_product / norm_product if norm_product > 0 else 0
        
        # Get match prediction
        is_match, confidence = calculate_match_score(cosine_sim)
        
        return {
            "model": "Pinecone + LLM",
            "prediction": "Match" if is_match else "Non-Match",
            "confidence": confidence,
            "cosine_similarity": round(float(cosine_sim), 4),
            "using_real_embeddings": True
        }
        
    except Exception as e:
        print(f"⚠️ Error in Pinecone+LLM prediction: {e}")
        return _fallback_prediction(resume_text, job_skills)


def _fallback_prediction(resume_text: str, job_skills: str) -> Dict:
    """
    Fallback prediction using TF-IDF when Pinecone/OpenAI is not available.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    try:
        # Simple TF-IDF based similarity
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
        resume_clean = resume_text.lower()
        job_clean = job_skills.lower()
        
        tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Also calculate word overlap for additional signal
        resume_words = set(resume_clean.split())
        job_words = set(job_clean.replace(',', ' ').split())
        intersection = len(resume_words.intersection(job_words))
        union = len(resume_words.union(job_words))
        jaccard = intersection / union if union > 0 else 0
        
        # Combine similarities
        combined_sim = (sim * 0.6) + (jaccard * 0.4)
        
        # Threshold for match
        match = combined_sim >= 0.15
        confidence = int(min(combined_sim * 200 + 30, 96))
        
        return {
            "model": "Pinecone + LLM",
            "prediction": "Match" if match else "Non-Match",
            "confidence": confidence,
            "using_real_embeddings": False,
            "fallback_reason": _initialization_error or "Using TF-IDF fallback"
        }
        
    except Exception as e:
        # Ultimate fallback
        resume_words = set(resume_text.lower().split())
        job_words = set(job_skills.lower().replace(',', ' ').split())
        overlap = len(resume_words.intersection(job_words))
        
        return {
            "model": "Pinecone + LLM",
            "prediction": "Match" if overlap >= 3 else "Non-Match",
            "confidence": min(overlap * 14 + 20, 96),
            "using_real_embeddings": False,
            "fallback_reason": str(e)
        }


def search_candidates_for_job(job_title: str, job_skills: str, top_k: int = 5) -> Dict:
    """
    Search for top matching candidates in Pinecone based on job description.
    
    Args:
        job_title: The job title
        job_skills: Required skills
        top_k: Number of top candidates to return
        
    Returns:
        Dictionary with candidates list and search metadata
    """
    if not _initialize_clients():
        return {
            "success": False,
            "error": _initialization_error or "Pinecone not initialized",
            "candidates": []
        }
    
    try:
        # Create job description text
        job_text = f"Job Title: {job_title}\nRequired Skills: {job_skills}"
        
        # Create embedding
        job_embedding = create_embedding(job_text)
        if job_embedding is None:
            return {
                "success": False,
                "error": "Failed to create job embedding",
                "candidates": []
            }
        
        # Search Pinecone
        matches = search_similar_resumes(job_embedding, top_k=top_k)
        
        # Format results
        candidates = []
        for i, match in enumerate(matches):
            is_match, confidence = calculate_match_score(match['score'])
            candidates.append({
                "rank": i + 1,
                "category": match['category'],
                "resume_id": match['resume_index'],
                "cosine_similarity": round(match['score'], 4),
                "match_percentage": confidence,
                "status": "Match" if is_match else "Non-Match",
                "text_preview": match['text_preview']
            })
        
        return {
            "success": True,
            "job_title": job_title,
            "candidates": candidates,
            "total_found": len(candidates)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "candidates": []
        }


def get_pinecone_status() -> Dict:
    """
    Get the current status of Pinecone connection.
    
    Returns:
        Dictionary with connection status and index info
    """
    if not _initialize_clients():
        return {
            "connected": False,
            "error": _initialization_error,
            "openai_configured": OPENAI_API_KEY is not None,
            "pinecone_configured": PINECONE_API_KEY is not None
        }
    
    try:
        stats = _pinecone_index.describe_index_stats()
        return {
            "connected": True,
            "index_name": INDEX_NAME,
            "total_vectors": stats['total_vector_count'],
            "dimension": stats.get('dimension', EMBEDDING_DIMENSIONS),
            "namespaces": list(stats.get('namespaces', {}).keys())
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }


# Export main function
__all__ = [
    'predict_match_pinecone_llm',
    'search_candidates_for_job', 
    'get_pinecone_status',
    'create_embedding'
]
