#!/usr/bin/env python3
"""
Job Description Matching Script
Performs cosine similarity search against resume embeddings in Pinecone
to find the top 3 matching candidates for a given job description.
"""

import os
import sys
import argparse
from typing import List, Dict, Optional
from openai import OpenAI
from pinecone import Pinecone
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'ai-hiring-portal')
EMBEDDING_MODEL = "text-embedding-3-small"  # Match the embedding script model
EMBEDDING_DIMENSIONS = 1024  # Match the dimensions from embedding script

class JobMatcher:
    def __init__(self):
        """Initialize the job matcher with OpenAI and Pinecone clients"""
        # Initialize OpenAI client
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Pinecone
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Connect to the index
        try:
            self.index = self.pc.Index(INDEX_NAME)
            print(f"✅ Connected to Pinecone index: {INDEX_NAME}")
            
            # Check index stats
            stats = self.index.describe_index_stats()
            print(f"📊 Index contains {stats['total_vector_count']} vectors")
            
        except Exception as e:
            print(f"❌ Error connecting to Pinecone index: {e}")
            raise

    def create_job_description_embedding(self, job_description: str) -> List[float]:
        """
        Create embedding for the job description using OpenAI API
        
        Args:
            job_description: The formatted job description text
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            print("🔄 Creating embedding for job description...")
            
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=job_description,
                dimensions=EMBEDDING_DIMENSIONS
            )
            
            print("✅ Job description embedding created successfully")
            return response.data[0].embedding
            
        except Exception as e:
            print(f"❌ Error creating job description embedding: {e}")
            raise

    def format_job_description(self, job_title: str, keywords: str, 
                             requirements: str = "", description: str = "") -> str:
        """
        Format job inputs into a comprehensive job description for embedding
        
        Args:
            job_title: The job title/position
            keywords: Important keywords and skills
            requirements: Job requirements (optional)
            description: Additional job description (optional)
            
        Returns:
            Formatted job description string
        """
        jd_parts = [f"Job Title: {job_title}"]
        
        if keywords:
            jd_parts.append(f"Required Skills and Keywords: {keywords}")
        
        if requirements:
            jd_parts.append(f"Requirements: {requirements}")
        
        if description:
            jd_parts.append(f"Job Description: {description}")
        
        return "\n\n".join(jd_parts)

    def search_matching_resumes(self, query_embedding: List[float], 
                              top_k: int = 3) -> List[Dict]:
        """
        Search for matching resumes using cosine similarity
        
        Args:
            query_embedding: The job description embedding vector
            top_k: Number of top matches to return
            
        Returns:
            List of dictionaries containing match results
        """
        try:
            print(f"🔍 Searching for top {top_k} matching resumes...")
            
            # Query Pinecone for similar vectors
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k * 5,  # Get more results to deduplicate by resume
                include_metadata=True
            )
            
            # Access matches - use safe attribute access
            matches = getattr(query_response, 'matches', [])
            
            if not matches:
                print("❌ No matches found")
                return []
            
            # Group results by resume to avoid duplicate resumes
            resume_matches = {}
            
            for match in matches:
                # Handle both dict and object-style access
                if isinstance(match, dict):
                    score = match['score']
                    metadata = match['metadata']
                    match_id = match['id']
                else:
                    score = match.score
                    metadata = match.metadata
                    match_id = match.id
                
                resume_index = metadata.get('resume_index')
                category = metadata.get('category')
                
                # Create a unique key for each resume
                resume_key = f"{category}_{resume_index}"
                
                # Keep only the best score for each resume
                if resume_key not in resume_matches or score > resume_matches[resume_key]['score']:
                    resume_matches[resume_key] = {
                        'score': score,
                        'category': category,
                        'resume_index': resume_index,
                        'text_preview': metadata.get('text', '')[:200] + "...",
                        'chunk_id': metadata.get('chunk_id'),
                        'match_id': match_id
                    }
            
            # Sort by score and return top matches
            sorted_matches = sorted(resume_matches.values(), 
                                  key=lambda x: x['score'], 
                                  reverse=True)
            
            return sorted_matches[:top_k]
            
        except Exception as e:
            print(f"❌ Error searching for matches: {e}")
            raise

    def calculate_match_percentage(self, cosine_score: float) -> float:
        """
        Convert cosine similarity score to a percentage
        Cosine similarity ranges from -1 to 1, where 1 is perfect match
        """
        # Convert to percentage (0-100)
        # Cosine similarity of 1.0 = 100% match
        # Cosine similarity of 0.0 = 50% match  
        # Cosine similarity of -1.0 = 0% match
        percentage = ((cosine_score + 1) / 2) * 100
        return round(percentage, 2)

    def display_results(self, matches: List[Dict], job_title: str):
        """
        Display the matching results in a formatted way
        
        Args:
            matches: List of match dictionaries
            job_title: The job title being matched
        """
        print("\n" + "="*80)
        print(f"🎯 TOP MATCHING CANDIDATES FOR: {job_title.upper()}")
        print("="*80)
        
        if not matches:
            print("❌ No suitable candidates found")
            return
        
        for i, match in enumerate(matches, 1):
            score_percentage = self.calculate_match_percentage(match['score'])
            
            print(f"\n🏆 RANK #{i}")
            print(f"📂 Category: {match['category']}")
            print(f"🆔 Resume ID: {match['resume_index']}")
            print(f"📊 Cosine Similarity: {match['score']:.4f}")
            print(f"📈 Match Percentage: {score_percentage}%")
            
            # Status based on match percentage
            if score_percentage >= 80:
                status = "✅ EXCELLENT MATCH"
            elif score_percentage >= 70:
                status = "🟢 GOOD MATCH"
            elif score_percentage >= 60:
                status = "🟡 MODERATE MATCH"
            else:
                status = "🔴 WEAK MATCH"
            
            print(f"🎯 Status: {status}")
            print(f"📝 Text Preview: {match['text_preview']}")
            print("-" * 60)
        
        # Summary
        best_match = matches[0]
        best_percentage = self.calculate_match_percentage(best_match['score'])
        print(f"\n🌟 BEST MATCH: {best_match['category']} (Resume #{best_match['resume_index']}) - {best_percentage}% match")

def main():
    """Main function to run the job matching"""
    parser = argparse.ArgumentParser(description="Find matching candidates for a job description")
    
    # Command line arguments
    parser.add_argument("--job-title", "-j", required=True, help="Job title/position")
    parser.add_argument("--keywords", "-k", required=True, help="Required skills and keywords")
    parser.add_argument("--requirements", "-r", default="", help="Job requirements")
    parser.add_argument("--description", "-d", default="", help="Additional job description")
    parser.add_argument("--top-k", "-t", type=int, default=3, help="Number of top matches to return")
    
    args = parser.parse_args()
    
    try:
        # Initialize the job matcher
        print("🚀 Initializing AI Hiring Portal Job Matcher...")
        matcher = JobMatcher()
        
        # Format the job description
        job_description = matcher.format_job_description(
            job_title=args.job_title,
            keywords=args.keywords,
            requirements=args.requirements,
            description=args.description
        )
        
        print("\n📋 Job Description:")
        print("-" * 40)
        print(job_description)
        print("-" * 40)
        
        # Create embedding for job description
        embedding = matcher.create_job_description_embedding(job_description)
        
        # Search for matching resumes
        matches = matcher.search_matching_resumes(embedding, top_k=args.top_k)
        
        # Display results
        matcher.display_results(matches, args.job_title)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def interactive_mode():
    """Interactive mode for easier use"""
    print("🤖 AI Hiring Portal - Interactive Job Matching")
    print("=" * 50)
    
    try:
        # Initialize matcher
        matcher = JobMatcher()
        
        # Get job details interactively
        print("\n📝 Please provide the job details:")
        job_title = input("Job Title: ").strip()
        keywords = input("Required Skills/Keywords: ").strip()
        requirements = input("Requirements (optional): ").strip()
        description = input("Additional Description (optional): ").strip()
        
        try:
            top_k = int(input("Number of matches to return (default 3): ").strip() or "3")
        except ValueError:
            top_k = 3
        
        if not job_title or not keywords:
            print("❌ Job title and keywords are required!")
            return
        
        # Format and process
        job_description = matcher.format_job_description(
            job_title=job_title,
            keywords=keywords,
            requirements=requirements,
            description=description
        )
        
        print(f"\n🔄 Processing job matching for: {job_title}")
        
        # Create embedding and search
        embedding = matcher.create_job_description_embedding(job_description)
        matches = matcher.search_matching_resumes(embedding, top_k=top_k)
        
        # Display results
        matcher.display_results(matches, job_title)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Check if command line arguments were provided
    if len(sys.argv) > 1:
        main()
    else:
        # Run in interactive mode
        interactive_mode()
