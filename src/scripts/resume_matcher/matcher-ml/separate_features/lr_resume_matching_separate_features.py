#!/usr/bin/env python3
"""
Resume Matching with Machine Learning - Separate Features Version
Using Logistic Regression with separate resume and job skill features
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from scipy.sparse import hstack

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

def prepare_features_separate(df):
    """Prepare separate features for resume and job skills"""
    print("\n🔧 Preparing separate features...")
    
    # Clean text data
    df['resume_text_clean'] = df['resume_text'].fillna("").str.lower()
    df['job_skill_text_clean'] = df['job_skill_text'].fillna("").str.lower()
    
    print(f"✅ Features prepared separately")
    print(f"   Resume text avg length: {df['resume_text_clean'].str.len().mean():.0f} chars")
    print(f"   Job skills avg length: {df['job_skill_text_clean'].str.len().mean():.0f} chars")
    
    return df

def train_model_separate_features(df):
    """Train model with separate resume and job skill features"""
    print("\n🤖 Training Model with Separate Features...")
    
    # Prepare separate feature sets
    X_resume = df['resume_text_clean']
    X_job = df['job_skill_text_clean']
    y = df['match_label']
    
    # Split the data
    X_resume_train, X_resume_test, X_job_train, X_job_test, y_train, y_test = train_test_split(
        X_resume, X_job, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set size: {len(X_resume_train)}")
    print(f"📊 Test set size: {len(X_resume_test)}")
    
    # Create separate TF-IDF vectorizers
    resume_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=2500,  # Split features between resume and job
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    job_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=2500,  # Split features between resume and job
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Transform text to TF-IDF vectors
    X_resume_train_tfidf = resume_vectorizer.fit_transform(X_resume_train)
    X_resume_test_tfidf = resume_vectorizer.transform(X_resume_test)
    
    X_job_train_tfidf = job_vectorizer.fit_transform(X_job_train)
    X_job_test_tfidf = job_vectorizer.transform(X_job_test)
    
    # Concatenate the feature matrices horizontally
    X_train_combined = hstack([X_resume_train_tfidf, X_job_train_tfidf])
    X_test_combined = hstack([X_resume_test_tfidf, X_job_test_tfidf])
    
    print(f"📊 Resume TF-IDF shape: {X_resume_train_tfidf.shape}")
    print(f"📊 Job skills TF-IDF shape: {X_job_train_tfidf.shape}")
    print(f"📊 Combined feature matrix shape: {X_train_combined.shape}")
    
    # Train logistic regression model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_combined, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_combined)
    y_pred_proba = model.predict_proba(X_test_combined)[:, 1]
    
    return model, resume_vectorizer, job_vectorizer, y_test, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, y_pred_proba):
    """Evaluate model performance"""
    print("\n📈 Model Evaluation Results (Separate Features):")
    print("=" * 60)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"🎯 AUC-ROC Score: {auc_score:.4f}")
    
    print(f"\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, auc_score

def predict_sample_separate(model, resume_vectorizer, job_vectorizer):
    """Test model with separate resume and job descriptions"""
    print("\n🧪 Testing with Separate Resume and Job Descriptions:")
    print("=" * 60)
    
    test_cases = [
        {
            "resume": "Experienced software engineer with expertise in Python, machine learning, and data analysis.",
            "job_skills": "Software development, Python, machine learning, data analysis",
            "expected": "High match"
        },
        {
            "resume": "Marketing professional with social media experience and content creation skills.",
            "job_skills": "Software development, Python, machine learning, data analysis", 
            "expected": "Low match"
        },
        {
            "resume": "Data scientist with R and Python experience in statistical analysis.",
            "job_skills": "Data science, Python, statistical analysis, machine learning",
            "expected": "High match"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        # Vectorize separately
        resume_tfidf = resume_vectorizer.transform([case["resume"].lower()])
        job_tfidf = job_vectorizer.transform([case["job_skills"].lower()])
        
        # Combine features
        combined_features = hstack([resume_tfidf, job_tfidf])
        
        # Predict
        prediction = model.predict(combined_features)[0]
        probability = model.predict_proba(combined_features)[0][1]
        
        print(f"\n🔍 Test Case {i}:")
        print(f"Resume: {case['resume']}")
        print(f"Job Skills: {case['job_skills']}")
        print(f"Expected: {case['expected']}")
        print(f"Prediction: {'✅ MATCH' if prediction == 1 else '❌ NO MATCH'}")
        print(f"Match Probability: {probability:.4f}")

def main():
    """Main function"""
    print("🤖 Resume Matching with Separate Features")
    print("Using Separate TF-IDF Vectors for Resume and Job Skills")
    print("=" * 70)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Prepare features
    df_processed = prepare_features_separate(df)
    
    # Train model
    model, resume_vec, job_vec, y_test, y_pred, y_pred_proba = train_model_separate_features(df_processed)
    
    # Evaluate
    accuracy, auc_score = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Test samples
    predict_sample_separate(model, resume_vec, job_vec)
    
    print(f"\n🎉 Separate Features Model Completed!")
    print(f"💡 Model Accuracy: {accuracy:.4f}")
    print(f"💡 AUC-ROC Score: {auc_score:.4f}")

if __name__ == "__main__":
    main()
