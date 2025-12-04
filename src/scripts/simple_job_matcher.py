#!/usr/bin/env python3
"""
Simple Job Matching Script
Easy-to-use command line interface for finding matching candidates
"""

import sys
import os

# Add the scripts directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from job_matching import JobMatcher

def main():
    print("🤖 AI Hiring Portal - Job Matching System")
    print("=" * 50)
    
    # Get job details from user
    print("\n📝 Enter job details:")
    job_title = input("Job Title: ").strip()
    
    if not job_title:
        print("❌ Job title is required!")
        return
    
    keywords = input("Required Skills/Keywords (comma-separated): ").strip()
    
    if not keywords:
        print("❌ Keywords are required!")
        return
    
    # Optional fields
    requirements = input("Additional Requirements (optional): ").strip()
    description = input("Job Description (optional): ").strip()
    
    # Number of matches
    try:
        num_matches = input("Number of matches to find (default 3): ").strip()
        num_matches = int(num_matches) if num_matches else 3
    except ValueError:
        num_matches = 3
    
    print(f"\n🔄 Searching for candidates matching: {job_title}...")
    
    try:
        # Initialize matcher
        matcher = JobMatcher()
        
        # Create job description
        job_description = matcher.format_job_description(
            job_title=job_title,
            keywords=keywords,
            requirements=requirements,
            description=description
        )
        
        # Get embedding and search
        embedding = matcher.create_job_description_embedding(job_description)
        matches = matcher.search_matching_resumes(embedding, top_k=num_matches)
        
        # Display results
        matcher.display_results(matches, job_title)
        
        # Save results to file
        if matches:
            save_results = input("\n💾 Save results to file? (y/n): ").strip().lower()
            if save_results == 'y':
                save_to_file(matches, job_title)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("📝 Make sure your .env file is configured correctly and Pinecone index exists.")

def save_to_file(matches, job_title):
    """Save matching results to a text file"""
    filename = f"job_matches_{job_title.replace(' ', '_').lower()}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Job Matching Results for: {job_title}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, match in enumerate(matches, 1):
                score_percentage = ((match['score'] + 1) / 2) * 100
                
                f.write(f"Rank #{i}\n")
                f.write(f"Category: {match['category']}\n")
                f.write(f"Resume ID: {match['resume_index']}\n")
                f.write(f"Match Percentage: {score_percentage:.2f}%\n")
                f.write(f"Preview: {match['text_preview']}\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"✅ Results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Could not save results: {e}")

if __name__ == "__main__":
    main()
