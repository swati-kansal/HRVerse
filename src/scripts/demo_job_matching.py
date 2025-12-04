#!/usr/bin/env python3
"""
Job Matching Demo Script
Demonstrates how the job matching system would work with sample data
"""

def demo_job_matching():
    """Demonstrate job matching functionality with sample data"""
    
    print("🤖 AI Hiring Portal - Job Matching Demo")
    print("=" * 60)
    
    # Sample job description
    job_title = "Senior Python Developer"
    keywords = "Python, Django, REST API, PostgreSQL, AWS, Docker"
    requirements = "5+ years experience, Bachelor's degree in CS"
    description = "We are looking for an experienced Python developer to join our backend team"
    
    print("\n📋 Sample Job Description:")
    print("-" * 40)
    print(f"Title: {job_title}")
    print(f"Keywords: {keywords}")
    print(f"Requirements: {requirements}")
    print(f"Description: {description}")
    print("-" * 40)
    
    # Simulate the matching process
    print("\n🔄 Processing Steps:")
    print("1. ✅ Formatting job description")
    print("2. ✅ Creating embedding with OpenAI API")
    print("3. ✅ Searching Pinecone vector database")
    print("4. ✅ Calculating cosine similarity scores")
    print("5. ✅ Ranking candidates by match percentage")
    
    # Sample matching results
    sample_matches = [
        {
            'category': 'INFORMATION-TECHNOLOGY',
            'resume_index': 42,
            'cosine_score': 0.8534,
            'text_preview': "Senior Software Engineer with 8 years experience in Python, Django, and AWS. Built scalable REST APIs serving millions of users...",
        },
        {
            'category': 'ENGINEERING',
            'resume_index': 156,
            'cosine_score': 0.8102,
            'text_preview': "Full-stack developer with expertise in Python/Django backend development. Experience with PostgreSQL, Docker containers...",
        },
        {
            'category': 'INFORMATION-TECHNOLOGY',
            'resume_index': 89,
            'cosine_score': 0.7845,
            'text_preview': "Python developer with 6 years experience. Strong background in web development, API design, and cloud platforms...",
        }
    ]
    
    # Display results
    print("\n" + "="*80)
    print(f"🎯 TOP MATCHING CANDIDATES FOR: {job_title.upper()}")
    print("="*80)
    
    for i, match in enumerate(sample_matches, 1):
        # Calculate match percentage
        score_percentage = ((match['cosine_score'] + 1) / 2) * 100
        
        print(f"\n🏆 RANK #{i}")
        print(f"📂 Category: {match['category']}")
        print(f"🆔 Resume ID: {match['resume_index']}")
        print(f"📊 Cosine Similarity: {match['cosine_score']:.4f}")
        print(f"📈 Match Percentage: {score_percentage:.2f}%")
        
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
        print(f"📝 Text Preview: {match['text_preview'][:100]}...")
        print("-" * 60)
    
    # Summary
    best_match = sample_matches[0]
    best_percentage = ((best_match['cosine_score'] + 1) / 2) * 100
    print(f"\n🌟 BEST MATCH: {best_match['category']} (Resume #{best_match['resume_index']}) - {best_percentage:.2f}% match")
    
    print(f"\n📊 Matching Statistics:")
    print(f"   • Total candidates evaluated: 1000+")
    print(f"   • Top matches returned: 3")
    print(f"   • Average match score: {sum(m['cosine_score'] for m in sample_matches) / len(sample_matches):.4f}")
    print(f"   • Search time: ~0.5 seconds")

def show_usage_instructions():
    """Show how to use the actual implementation"""
    
    print("\n" + "="*80)
    print("🚀 HOW TO USE THE ACTUAL IMPLEMENTATION")
    print("="*80)
    
    print("\n1️⃣ **Setup Environment:**")
    print("   • Create virtual environment: python3 -m venv venv")
    print("   • Activate it: source venv/bin/activate")
    print("   • Install packages: pip install -r requirements.txt")
    print("   • Configure .env file with your API keys")
    
    print("\n2️⃣ **Embed Resumes (First Time Only):**")
    print("   python src/scripts/embed_resumes_to_pinecone.py")
    
    print("\n3️⃣ **Find Matching Candidates:**")
    print("   • Interactive mode: python src/scripts/simple_job_matcher.py")
    print("   • Command line: python src/scripts/job_matching.py --job-title \"Developer\" --keywords \"Python\"")
    
    print("\n4️⃣ **Example Command:**")
    print("""   python src/scripts/job_matching.py \\
       --job-title "Data Scientist" \\
       --keywords "Python, Machine Learning, Pandas" \\
       --requirements "PhD preferred" \\
       --top-k 5""")
    
    print("\n📋 **Required Environment Variables:**")
    print("   OPENAI_API_KEY=your_openai_key")
    print("   PINECONE_API_KEY=your_pinecone_key")
    print("   PINECONE_INDEX_NAME=ai-hiring-portal")
    
    print("\n🎯 **What You Get:**")
    print("   • Top matching candidates ranked by similarity")
    print("   • Match percentage scores (0-100%)")
    print("   • Resume previews and categories")
    print("   • Status indicators (Excellent/Good/Moderate/Weak)")
    print("   • Option to save results to file")

def interactive_demo():
    """Interactive demo mode"""
    
    print("\n🤖 Interactive Demo Mode")
    print("=" * 40)
    
    try:
        job_title = input("Enter a job title to search for: ").strip()
        keywords = input("Enter required skills/keywords: ").strip()
        
        if not job_title or not keywords:
            print("❌ Both job title and keywords are required for demonstration")
            return
        
        print(f"\n🔍 Searching for candidates matching: {job_title}")
        print(f"📝 Required skills: {keywords}")
        print("\n⏳ In a real implementation, this would:")
        print("   1. Create embedding for your job description")
        print("   2. Search against 1000+ resume embeddings")
        print("   3. Return top matches with similarity scores")
        print("   4. Provide actionable candidate recommendations")
        
        # Simulate processing time
        import time
        for i in range(3):
            print("   🔄 Processing..." if i == 0 else "   🔄 Calculating similarities..." if i == 1 else "   ✅ Complete!")
            time.sleep(1)
        
        print(f"\n🎉 Found 3 excellent candidates for {job_title} position!")
        print("   📊 Match scores: 92.7%, 90.5%, 88.3%")
        print("   ⏱️  Total search time: 1.2 seconds")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo cancelled")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        demo_job_matching()
        show_usage_instructions()
        
        choice = input("\n🤔 Try interactive demo? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_demo()
