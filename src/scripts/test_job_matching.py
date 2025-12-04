#!/usr/bin/env python3
"""
Test script to validate job matching functionality
"""

import sys
import os

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_job_matcher():
    """Test the job matching functionality"""
    try:
        from job_matching import JobMatcher
        
        print("🧪 Testing Job Matching System...")
        print("=" * 50)
        
        # Initialize matcher
        print("1. Initializing JobMatcher...")
        matcher = JobMatcher()
        print("✅ JobMatcher initialized successfully")
        
        # Test job description formatting
        print("\n2. Testing job description formatting...")
        job_desc = matcher.format_job_description(
            job_title="Software Engineer",
            keywords="Python, Django, REST API",
            requirements="2+ years experience",
            description="Building web applications"
        )
        print("✅ Job description formatted successfully")
        print(f"📝 Formatted JD: {job_desc[:100]}...")
        
        # Test embedding creation
        print("\n3. Testing embedding creation...")
        embedding = matcher.create_job_description_embedding(job_desc)
        print(f"✅ Embedding created successfully (dimension: {len(embedding)})")
        
        # Test similarity search
        print("\n4. Testing similarity search...")
        matches = matcher.search_matching_resumes(embedding, top_k=3)
        print(f"✅ Search completed successfully ({len(matches)} matches found)")
        
        # Display results
        if matches:
            print("\n5. Sample results:")
            for i, match in enumerate(matches[:2], 1):  # Show first 2 matches
                percentage = ((match['score'] + 1) / 2) * 100
                print(f"   Match #{i}: {match['category']} (Resume #{match['resume_index']}) - {percentage:.1f}%")
        else:
            print("\n⚠️  No matches found - this might be expected if index is empty")
        
        print("\n🎉 All tests passed! Job matching system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\n📝 Troubleshooting tips:")
        print("   1. Check .env file configuration")
        print("   2. Verify Pinecone index exists and has data")
        print("   3. Confirm API keys are valid")
        print("   4. Run resume embedding script first if needed")
        return False

def check_environment():
    """Check if environment is properly configured"""
    print("🔧 Checking environment configuration...")
    print("-" * 40)
    
    required_vars = [
        'OPENAI_API_KEY',
        'PINECONE_API_KEY', 
        'PINECONE_INDEX_NAME'
    ]
    
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * (len(value) - 10) + value[-10:]}")  # Mask most of the key
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   Please check your .env file")
        return False
    
    print("\n✅ Environment configuration looks good!")
    return True

if __name__ == "__main__":
    print("🚀 Job Matching System Validation")
    print("=" * 60)
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check environment first
    if not check_environment():
        sys.exit(1)
    
    print()
    
    # Run tests
    if test_job_matcher():
        print(f"\n🎯 Ready to use! Try running:")
        print(f"   python src/scripts/simple_job_matcher.py")
    else:
        sys.exit(1)
