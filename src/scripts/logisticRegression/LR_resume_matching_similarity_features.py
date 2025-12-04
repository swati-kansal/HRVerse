#!/usr/bin/env python3
"""
Resume Matching with Machine Learning - Similarity Features Version
Using cosine similarity between resume and job descriptions as features
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Load the synthetic dataset from CSV file"""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'storage', 'synthetic_resume_job_matching_1000_records.csv')
    
    try:
        print("📥 Loading synthetic dataset...")
        df = pd.read_csv(csv_path)
        print(f"✅ Dataset loaded successfully: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def create_similarity_features(df):
    """Create similarity-based features between resume and job descriptions"""
    print("\n🔧 Creating similarity-based features...")
    
    # Clean text
    df['resume_clean'] = df['resume_text'].fillna("").str.lower()
    df['job_clean'] = df['job_skill_text'].fillna("").str.lower()
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2
    )
    
    # Combine all text to fit vectorizer
    all_text = list(df['resume_clean']) + list(df['job_clean'])
    vectorizer.fit(all_text)
    
    # Transform resume and job texts separately
    resume_vectors = vectorizer.transform(df['resume_clean'])
    job_vectors = vectorizer.transform(df['job_clean'])
    
    # Calculate similarity features
    similarity_scores = []
    resume_lengths = []
    job_lengths = []
    
    for i in range(len(df)):
        # Cosine similarity between resume and job
        similarity = cosine_similarity(
            resume_vectors[i:i+1], 
            job_vectors[i:i+1]
        )[0][0]
        similarity_scores.append(similarity)
        
        # Text length features
        resume_lengths.append(len(df['resume_clean'].iloc[i]))
        job_lengths.append(len(df['job_clean'].iloc[i]))
    
    # Create feature dataframe
    feature_df = pd.DataFrame({
        'cosine_similarity': similarity_scores,
        'resume_length': resume_lengths,
        'job_length': job_lengths,
        'length_ratio': np.array(resume_lengths) / (np.array(job_lengths) + 1),  # +1 to avoid division by zero
    })
    
    print(f"✅ Similarity features created:")
    print(f"   Average cosine similarity: {np.mean(similarity_scores):.4f}")
    print(f"   Average resume length: {np.mean(resume_lengths):.0f} chars")
    print(f"   Average job description length: {np.mean(job_lengths):.0f} chars")
    
    return feature_df, vectorizer

def train_similarity_model(feature_df, labels):
    """Train model using similarity features"""
    print("\n🤖 Training Model with Similarity Features...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Training set size: {len(X_train)}")
    print(f"📊 Test set size: {len(X_test)}")
    print(f"📊 Feature columns: {list(feature_df.columns)}")
    
    # Train model
    model = LogisticRegression(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Show feature importance
    print(f"\n🔍 Feature Importance (Coefficients):")
    for feature, coef in zip(feature_df.columns, model.coef_[0]):
        print(f"   {feature}: {coef:.4f}")
    
    return model, X_test, y_test, y_pred, y_pred_proba

def predict_similarity_sample(model, vectorizer):
    """Test the similarity-based model"""
    print("\n🧪 Testing Similarity-Based Model:")
    print("=" * 50)
    
    test_cases = [
        {
            "resume": "Python developer with machine learning experience and data analysis skills",
            "job": "Python, machine learning, data analysis, software development",
            "expected": "High similarity"
        },
        {
            "resume": "Marketing manager with social media and advertising experience",
            "job": "Python, machine learning, data analysis, software development",
            "expected": "Low similarity"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        # Calculate similarity features for this case
        resume_vec = vectorizer.transform([case["resume"].lower()])
        job_vec = vectorizer.transform([case["job"].lower()])
        
        similarity = cosine_similarity(resume_vec, job_vec)[0][0]
        resume_len = len(case["resume"])
        job_len = len(case["job"])
        length_ratio = resume_len / (job_len + 1)
        
        # Create feature vector
        features = pd.DataFrame({
            'cosine_similarity': [similarity],
            'resume_length': [resume_len],
            'job_length': [job_len],
            'length_ratio': [length_ratio]
        })
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        print(f"\n🔍 Test Case {i}:")
        print(f"Resume: {case['resume']}")
        print(f"Job: {case['job']}")
        print(f"Cosine Similarity: {similarity:.4f}")
        print(f"Expected: {case['expected']}")
        print(f"Prediction: {'✅ MATCH' if prediction == 1 else '❌ NO MATCH'}")
        print(f"Match Probability: {probability:.4f}")

def main():
    """Main function for similarity-based approach"""
    print("🤖 Resume Matching with Similarity Features")
    print("Using Cosine Similarity and Text Statistics")
    print("=" * 60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create features
    features, vectorizer = create_similarity_features(df)
    
    # Train model
    model, X_test, y_test, y_pred, y_pred_proba = train_similarity_model(features, df['match_label'])
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📈 Similarity Model Results:")
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"🎯 AUC-ROC Score: {auc_score:.4f}")
    print(f"\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Test samples
    predict_similarity_sample(model, vectorizer)
    
    print(f"\n🎉 Similarity-Based Model Completed!")

if __name__ == "__main__":
    main()
