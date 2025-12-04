#!/usr/bin/env python3
"""
Resume Matching with Machine Learning
Using Logistic Regression for resume-job matching with synthetic dataset
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load the synthetic dataset from CSV file"""
    # Get the path to the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'storage', 'synthetic_resume_job_matching_1000_records.csv')
    
    try:
        print("📥 Loading synthetic dataset...")
        df = pd.read_csv(csv_path)
        print(f" Dataset loaded successfully: {df.shape}")
        print(" Dataset columns:", list(df.columns))
        
        # Display basic statistics
        print(f"\n📈 Dataset Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Match labels distribution:")
        print(df['match_label'].value_counts())
        print(f"Bias types:")
        print(df['bias_type'].value_counts())
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find CSV file at {csv_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

# Load the dataset
df = load_data()

if df is None:
    print("❌ Failed to load dataset. Exiting...")
    exit(1)

print("\n📋 Dataset preview:")
print(df.head())
def prepare_features(df):
    """Prepare features for machine learning model"""
    print("\n🔧 Preparing features...")
    
    # Combine resume text and job skills for feature creation
    df['combined_text'] = df['resume_text'].fillna("") + " " + df['job_skill_text'].fillna("")
    
    # Clean and preprocess text
    df['combined_text'] = df['combined_text'].str.lower()
    
    print(f"✅ Features prepared. Combined text length: {df['combined_text'].str.len().mean():.0f} chars on average")
    
    return df

def train_model(df):
    """Train logistic regression model"""
    print("\n🤖 Training Logistic Regression Model...")
    
    # Prepare features
    X = df['combined_text']
    y = df['match_label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set size: {len(X_train)}")
    print(f"📊 Test set size: {len(X_test)}")
    print(f"📊 Training set positive rate: {y_train.mean():.3f}")
    print(f"📊 Test set positive rate: {y_test.mean():.3f}")
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2),  # Use both unigrams and bigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"📊 TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    
    # Train logistic regression model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    return model, vectorizer, X_test, y_test, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, y_pred_proba):
    """Evaluate model performance"""
    print("\n📈 Model Evaluation Results:")
    print("=" * 50)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"🎯 AUC-ROC Score: {auc_score:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(cm)
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, auc_score

def predict_sample_resumes(model, vectorizer):
    """Test the model with sample resumes"""
    print("\n🧪 Testing with Sample Resumes:")
    print("=" * 50)
    
    sample_resumes = [
        {
            "text": "Experienced software engineer with expertise in Python, machine learning, and data analysis. Strong background in AI and statistical modeling.",
            "expected": "High match probability"
        },
        {
            "text": "Marketing professional with social media experience and content creation skills. No technical background.",
            "expected": "Low match probability"
        },
        {
            "text": "Data scientist with R and Python experience. Skilled in statistical analysis and machine learning algorithms.",
            "expected": "High match probability"
        },
        {
            "text": "Graphic designer with Adobe Creative Suite skills. Experience in branding and visual design.",
            "expected": "Low match probability"
        }
    ]
    
    for i, sample in enumerate(sample_resumes, 1):
        # Vectorize the sample text
        sample_tfidf = vectorizer.transform([sample["text"]])
        
        # Get prediction and probability
        prediction = model.predict(sample_tfidf)[0]
        probability = model.predict_proba(sample_tfidf)[0][1]
        
        print(f"\n🔍 Sample {i}:")
        print(f"Text: {sample['text'][:100]}...")
        print(f"Expected: {sample['expected']}")
        print(f"Prediction: {'✅ MATCH' if prediction == 1 else '❌ NO MATCH'}")
        print(f"Match Probability: {probability:.4f}")

def get_feature_importance(model, vectorizer, top_n=20):
    """Display most important features for the model"""
    print(f"\n🔍 Top {top_n} Most Important Features:")
    print("=" * 50)
    
    # Get feature names and coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Get top positive and negative features
    top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_negative_idx = np.argsort(coefficients)[:top_n]
    
    print("🔸 Features indicating MATCH:")
    for idx in top_positive_idx:
        print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
    
    print("\n🔸 Features indicating NO MATCH:")
    for idx in top_negative_idx:
        print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

def main():
    """Main function to run the ML resume matching pipeline"""
    print("🤖 Resume Matching with Machine Learning")
    print("Using Logistic Regression with TF-IDF Features")
    print("=" * 60)
    
    # Prepare features
    df_processed = prepare_features(df)
    
    # Train model
    model, vectorizer, X_test, y_test, y_pred, y_pred_proba = train_model(df_processed)
    
    # Evaluate model
    accuracy, auc_score = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Show feature importance
    get_feature_importance(model, vectorizer)
    
    # Test with sample resumes
    predict_sample_resumes(model, vectorizer)
    
    print(f"\n🎉 ML Pipeline Completed Successfully!")
    print(f"💡 Model Accuracy: {accuracy:.4f}")
    print(f"💡 AUC-ROC Score: {auc_score:.4f}")
    
    return model, vectorizer

if __name__ == "__main__":
    trained_model, trained_vectorizer = main()
