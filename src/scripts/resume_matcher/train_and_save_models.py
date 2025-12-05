#!/usr/bin/env python3
"""
Train and Save Resume Matching Models
Trains three Logistic Regression models and saves them for use in the Flask app
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', '..', 'storage', 'resume_matcher', 'synthetic_resume_job_matching_1000_records.csv')
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, '..', '..', '..', 'storage', 'resume_matcher', 'trained_lr_models.pkl')


def load_data():
    """Load the synthetic dataset"""
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} records")
    print(f"   Match distribution: {df['match_label'].value_counts().to_dict()}")
    return df


def train_separate_features_model(df):
    """
    Train Logistic Regression with SEPARATE features for resume and job skills.
    Each text is vectorized independently and then concatenated.
    """
    print("\n" + "="*60)
    print("🤖 Training Model 1: SEPARATE FEATURES")
    print("="*60)
    
    # Clean text
    df['resume_clean'] = df['resume_text'].fillna("").str.lower()
    df['job_clean'] = df['job_skill_text'].fillna("").str.lower()
    
    # Split data
    X_resume = df['resume_clean']
    X_job = df['job_clean']
    y = df['match_label']
    
    X_resume_train, X_resume_test, X_job_train, X_job_test, y_train, y_test = train_test_split(
        X_resume, X_job, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create separate TF-IDF vectorizers
    resume_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    job_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    # Transform
    X_resume_train_tfidf = resume_vectorizer.fit_transform(X_resume_train)
    X_resume_test_tfidf = resume_vectorizer.transform(X_resume_test)
    
    X_job_train_tfidf = job_vectorizer.fit_transform(X_job_train)
    X_job_test_tfidf = job_vectorizer.transform(X_job_test)
    
    # Combine features
    X_train = hstack([X_resume_train_tfidf, X_job_train_tfidf])
    X_test = hstack([X_resume_test_tfidf, X_job_test_tfidf])
    
    print(f"   Resume features: {X_resume_train_tfidf.shape[1]}")
    print(f"   Job features: {X_job_train_tfidf.shape[1]}")
    print(f"   Combined features: {X_train.shape[1]}")
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n📊 Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Non-Match', 'Match']))
    
    return {
        'model': model,
        'resume_vectorizer': resume_vectorizer,
        'job_vectorizer': job_vectorizer,
        'accuracy': accuracy,
        'auc': auc
    }


def train_combined_features_model(df):
    """
    Train Logistic Regression with COMBINED features.
    Resume and job text are concatenated before vectorization.
    """
    print("\n" + "="*60)
    print("🤖 Training Model 2: COMBINED FEATURES")
    print("="*60)
    
    # Combine text
    df['combined_text'] = (df['resume_text'].fillna("") + " [SEP] " + df['job_skill_text'].fillna("")).str.lower()
    
    X = df['combined_text']
    y = df['match_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Single vectorizer for combined text
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   Combined features: {X_train_tfidf.shape[1]}")
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n📊 Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Non-Match', 'Match']))
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'auc': auc
    }


def train_similarity_features_model(df):
    """
    Train Logistic Regression with SIMILARITY features.
    Uses cosine similarity between resume and job TF-IDF vectors as features.
    """
    print("\n" + "="*60)
    print("🤖 Training Model 3: SIMILARITY FEATURES")
    print("="*60)
    
    # Clean text
    df['resume_clean'] = df['resume_text'].fillna("").str.lower()
    df['job_clean'] = df['job_skill_text'].fillna("").str.lower()
    
    # Create a single vectorizer for both (to ensure same feature space)
    all_text = pd.concat([df['resume_clean'], df['job_clean']])
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    vectorizer.fit(all_text)
    
    # Transform texts
    resume_vectors = vectorizer.transform(df['resume_clean'])
    job_vectors = vectorizer.transform(df['job_clean'])
    
    # Calculate similarity features
    similarities = []
    for i in range(len(df)):
        sim = cosine_similarity(resume_vectors[i], job_vectors[i])[0][0]
        similarities.append(sim)
    
    df['similarity'] = similarities
    
    # Create additional features
    # Jaccard similarity on words
    def jaccard_similarity(text1, text2):
        set1 = set(text1.split())
        set2 = set(text2.split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    df['jaccard'] = df.apply(lambda row: jaccard_similarity(row['resume_clean'], row['job_clean']), axis=1)
    
    # Skill overlap count
    def skill_overlap(resume, job_skills):
        resume_words = set(resume.lower().split())
        job_words = set(job_skills.lower().replace(',', ' ').split())
        return len(resume_words.intersection(job_words))
    
    df['skill_overlap'] = df.apply(lambda row: skill_overlap(row['resume_clean'], row['job_clean']), axis=1)
    
    # Normalize skill overlap
    df['skill_overlap_norm'] = df['skill_overlap'] / df['job_clean'].apply(lambda x: len(x.split()))
    
    # Features for model
    X = df[['similarity', 'jaccard', 'skill_overlap', 'skill_overlap_norm']].values
    y = df['match_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Similarity features: {X.shape[1]}")
    print(f"   Feature names: similarity, jaccard, skill_overlap, skill_overlap_norm")
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n📊 Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Non-Match', 'Match']))
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'auc': auc
    }


def main():
    print("🚀 Training Resume Matching Models")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Train all three models
    separate_model = train_separate_features_model(df.copy())
    combined_model = train_combined_features_model(df.copy())
    similarity_model = train_similarity_features_model(df.copy())
    
    # Package all models
    models_package = {
        'separate_features': separate_model,
        'combined_features': combined_model,
        'similarity_features': similarity_model,
        'metadata': {
            'training_samples': len(df),
            'match_ratio': df['match_label'].mean()
        }
    }
    
    # Save to pickle file
    print("\n" + "="*60)
    print("💾 Saving models...")
    
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(models_package, f)
    
    print(f"✅ Models saved to: {MODEL_OUTPUT_PATH}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':<12} {'AUC-ROC':<12}")
    print("-"*50)
    print(f"{'Separate Features':<25} {separate_model['accuracy']:.4f}       {separate_model['auc']:.4f}")
    print(f"{'Combined Features':<25} {combined_model['accuracy']:.4f}       {combined_model['auc']:.4f}")
    print(f"{'Similarity Features':<25} {similarity_model['accuracy']:.4f}       {similarity_model['auc']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
