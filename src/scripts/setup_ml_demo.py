#!/usr/bin/env python3
"""
Setup and Demo Script for ML-based Resume Matching
Installs required packages and demonstrates functionality
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages for ML functionality"""
    packages = [
        'scikit-learn>=1.5.0',
        'joblib>=1.3.0', 
        'datasets>=2.0.0',
        'python-dotenv',
        'pandas>=2.0.0',
        'numpy>=1.20.0'
    ]
    
    print("📦 Installing required packages...")
    
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def test_imports():
    """Test if all required imports work"""
    print("\n🧪 Testing imports...")
    
    try:
        import sklearn
        print("   ✅ scikit-learn imported successfully")
        
        import joblib
        print("   ✅ joblib imported successfully")
        
        import datasets
        print("   ✅ datasets imported successfully")
        
        import pandas as pd
        print("   ✅ pandas imported successfully")
        
        import numpy as np
        print("   ✅ numpy imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def demo_ml_functionality():
    """Demo the ML functionality with sample data"""
    print("\n🎮 Demo: ML-based Resume Matching")
    print("=" * 50)
    
    try:
        # Create sample data
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import accuracy_score, classification_report
        
        # Sample resume data
        sample_data = {
            'resume_text': [
                "Software Engineer with 5 years Python experience. Django, REST APIs, AWS cloud. Bachelor's Computer Science.",
                "Junior Developer fresh graduate. Java programming, Spring framework, MySQL database. Bachelor's degree.",
                "Senior Data Scientist with PhD. Machine Learning, TensorFlow, Python, R. 8 years experience analytics.",
                "Frontend Developer specializing in React, JavaScript, HTML/CSS. 3 years experience building web apps.",
                "DevOps Engineer with Docker, Kubernetes, Jenkins. 4 years infrastructure automation experience.",
                "QA Engineer with automation testing experience. Selenium, pytest, 2 years software testing.",
                "Project Manager with MBA. Agile, Scrum methodologies. 6 years managing software development teams.",
                "UI/UX Designer with Figma, Adobe Creative Suite. 3 years designing mobile and web interfaces."
            ],
            'job_description': [
                "Senior Python Developer needed. Django, AWS, REST APIs required. 5+ years experience.",
                "Junior Java Developer position. Spring framework knowledge preferred. Entry level.",
                "Data Scientist role. Machine Learning expertise required. PhD preferred. Python/R skills.",
                "Frontend Developer needed. React, JavaScript experience required. 2+ years experience.",
                "DevOps position. Docker, Kubernetes experience. Infrastructure automation skills needed.",
                "QA Automation Engineer. Testing frameworks experience. Selenium knowledge preferred.",
                "Technical Project Manager. Software development background. Agile experience required.",
                "UX Designer role. Design tools proficiency. Mobile app design experience preferred."
            ],
            'match_label': [1, 1, 1, 1, 1, 1, 1, 1]  # All matches for this demo
        }
        
        # Add some mismatched examples
        sample_data['resume_text'].extend([
            "Marketing Manager with social media expertise. 5 years digital marketing campaigns.",
            "Accountant with CPA certification. Financial analysis, Excel, SAP experience.",
            "Retail Sales Associate. Customer service, cash handling, inventory management.",
            "Chef with culinary arts degree. Fine dining experience, menu planning, kitchen management."
        ])
        sample_data['job_description'].extend([
            "Software Engineer position. Programming skills required.",
            "Software Engineer position. Programming skills required.", 
            "Software Engineer position. Programming skills required.",
            "Software Engineer position. Programming skills required."
        ])
        sample_data['match_label'].extend([0, 0, 0, 0])  # Mismatches
        
        df = pd.DataFrame(sample_data)
        
        print(f"📊 Sample Dataset: {len(df)} examples")
        print(f"   - Matches: {sum(df['match_label'])}")
        print(f"   - Mismatches: {len(df) - sum(df['match_label'])}")
        
        # Feature extraction
        print("\n⚙️ Feature Engineering...")
        
        # Simple features
        df['resume_length'] = df['resume_text'].str.len()
        df['word_count'] = df['resume_text'].str.split().str.len()
        df['experience_mentioned'] = df['resume_text'].str.contains('experience|years', case=False).astype(int)
        df['degree_mentioned'] = df['resume_text'].str.contains('bachelor|master|phd', case=False).astype(int)
        
        # TF-IDF for text features
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        text_features = vectorizer.fit_transform(df['resume_text']).toarray()
        
        # Combine features
        numeric_features = df[['resume_length', 'word_count', 'experience_mentioned', 'degree_mentioned']].values
        X = np.hstack([numeric_features, text_features])
        y = df['match_label'].values
        
        print(f"   Features shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        print("\n🚂 Training Logistic Regression...")
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"✅ Training completed!")
        print(f"   Training Accuracy: {train_acc:.2%}")
        print(f"   Test Accuracy: {test_acc:.2%}")
        
        print(f"\n📈 Detailed Results:")
        print(classification_report(y_test, test_pred, target_names=['Mismatch', 'Match']))
        
        # Demo prediction
        print(f"\n🎯 Demo Prediction:")
        sample_resume = "Senior Python Developer with Django and AWS experience. 7 years backend development."
        
        # Extract features for sample
        sample_df = pd.DataFrame({'resume_text': [sample_resume]})
        sample_df['resume_length'] = sample_df['resume_text'].str.len()
        sample_df['word_count'] = sample_df['resume_text'].str.split().str.len()
        sample_df['experience_mentioned'] = sample_df['resume_text'].str.contains('experience|years', case=False).astype(int)
        sample_df['degree_mentioned'] = sample_df['resume_text'].str.contains('bachelor|master|phd', case=False).astype(int)
        
        sample_text_features = vectorizer.transform(sample_df['resume_text']).toarray()
        sample_numeric = sample_df[['resume_length', 'word_count', 'experience_mentioned', 'degree_mentioned']].values
        sample_X = np.hstack([sample_numeric, sample_text_features])
        
        prediction = model.predict(sample_X)[0]
        probability = model.predict_proba(sample_X)[0]
        
        print(f"   Resume: {sample_resume[:60]}...")
        print(f"   Prediction: {'MATCH' if prediction == 1 else 'MISMATCH'}")
        print(f"   Match Probability: {probability[1]:.2%}")
        print(f"   Confidence: {max(probability):.2%}")
        
        print(f"\n🎉 ML Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup and demo function"""
    print("🚀 ML Resume Matching Setup & Demo")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ is required for optimal compatibility")
        return
    
    # Install packages
    choice = input("\n📦 Install required packages? (y/n): ").strip().lower()
    if choice == 'y':
        if not install_packages():
            print("❌ Package installation failed")
            return
    
    # Test imports
    if not test_imports():
        print("❌ Import tests failed. Try installing packages first.")
        return
    
    # Run demo
    choice = input("\n🎮 Run ML functionality demo? (y/n): ").strip().lower()
    if choice == 'y':
        if demo_ml_functionality():
            print("\n✨ Setup and demo completed successfully!")
            print("\n🎯 Next steps:")
            print("   1. Run the full ML script: python src/scripts/ml_resume_matching.py")
            print("   2. Load real Hugging Face dataset for training")
            print("   3. Compare with vector similarity approach")
        else:
            print("❌ Demo failed")
    else:
        print("\n✅ Setup completed! Ready to run ML resume matching.")

if __name__ == "__main__":
    main()
