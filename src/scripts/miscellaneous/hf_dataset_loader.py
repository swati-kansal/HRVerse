#!/usr/bin/env python3
"""
Resume Dataset Loader using Hugging Face Datasets
Downloads and processes the netsol/resume-score-details dataset for ML training
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import re

# Check for required libraries
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️  datasets library not available. Please install with: pip install datasets")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available. Please install with: pip install scikit-learn")

class HuggingFaceResumeDatasetLoader:
    def __init__(self):
        """Initialize the dataset loader"""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library is required. Please install with: pip install datasets")
        
        self.dataset = None
        self.processed_df = None
        self.feature_names = []
        self.label_mapping = {'matched': 1, 'mismatched': 0, 'match': 1, 'mismatch': 0}
        
    def load_dataset(self, dataset_name: str = "netsol/resume-score-details"):
        """Load the dataset from Hugging Face Hub"""
        try:
            print(f"📥 Loading dataset: {dataset_name}")
            print("🔄 This may take a moment for first download...")
            
            # Load the dataset
            self.dataset = load_dataset(dataset_name)
            
            print("✅ Dataset loaded successfully!")
            print(f"📊 Available splits: {list(self.dataset.keys())}")
            
            # Get the main split (usually 'train')
            main_split = 'train' if 'train' in self.dataset else list(self.dataset.keys())[0]
            df = pd.DataFrame(self.dataset[main_split])
            
            print(f"📋 Dataset shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            print(f"📋 First few rows:")
            print(df.head())
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            if "ConnectionError" in str(e) or "Couldn't reach" in str(e):
                print("🌐 Network connectivity issue. Please check your internet connection.")
                print("💡 You can also try running this script later when connectivity is better.")
            raise
    
    def inspect_dataset(self, df: pd.DataFrame):
        """Inspect the dataset structure and identify key fields"""
        print("\n🔍 Dataset Inspection")
        print("=" * 50)
        
        print(f"📊 Dataset Info:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        print(f"\n📝 Column Details:")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            
            print(f"   {col:25s} | Type: {str(dtype):10s} | Non-null: {non_null:4d} | Null: {null_count:4d}")
            
            # Show sample values for text columns
            if dtype == 'object':
                sample_values = df[col].dropna().head(3).tolist()
                print(f"      Sample values: {sample_values}")
        
        # Try to identify key fields
        print(f"\n🎯 Field Analysis:")
        
        # Look for resume text fields
        resume_fields = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['resume', 'cv', 'text', 'content'])]
        if resume_fields:
            print(f"   📄 Resume text fields: {resume_fields}")
        
        # Look for job description fields
        job_fields = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['job', 'jd', 'description', 'position', 'role'])]
        if job_fields:
            print(f"   💼 Job description fields: {job_fields}")
        
        # Look for skills fields
        skill_fields = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['skill', 'competenc', 'abilit'])]
        if skill_fields:
            print(f"   🛠️  Skills fields: {skill_fields}")
        
        # Look for label/target fields
        label_fields = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['match', 'label', 'target', 'score', 'rating'])]
        if label_fields:
            print(f"   🎯 Label/target fields: {label_fields}")
            
            # Show distribution of label fields
            for field in label_fields:
                print(f"      {field} distribution:")
                print(f"        {df[field].value_counts().to_dict()}")
        
        # Look for experience fields
        exp_fields = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['experience', 'exp', 'year', 'seniority'])]
        if exp_fields:
            print(f"   📈 Experience fields: {exp_fields}")
        
        # Look for education fields
        edu_fields = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['education', 'degree', 'qualification', 'study'])]
        if edu_fields:
            print(f"   🎓 Education fields: {edu_fields}")
        
        return {
            'resume_fields': resume_fields,
            'job_fields': job_fields,
            'skill_fields': skill_fields,
            'label_fields': label_fields,
            'experience_fields': exp_fields,
            'education_fields': edu_fields
        }
    
    def prepare_ml_data(self, df: pd.DataFrame, field_mapping: Dict) -> pd.DataFrame:
        """Prepare the dataset for machine learning"""
        print("\n🔧 Preparing data for ML training...")
        
        # Create a cleaned dataset
        ml_df = pd.DataFrame()
        
        # Extract text fields
        if field_mapping['resume_fields']:
            resume_col = field_mapping['resume_fields'][0]
            ml_df['resume_text'] = df[resume_col].fillna('')
        
        if field_mapping['job_fields']:
            job_col = field_mapping['job_fields'][0]
            ml_df['job_description'] = df[job_col].fillna('')
        
        # Extract label/target
        if field_mapping['label_fields']:
            label_col = field_mapping['label_fields'][0]
            ml_df['label_raw'] = df[label_col]
            
            # Map labels to binary
            ml_df['is_match'] = ml_df['label_raw'].map(self.label_mapping)
            
            # If mapping failed, try to infer from text
            if ml_df['is_match'].isnull().all():
                print("🔄 Attempting to infer labels from text...")
                ml_df['is_match'] = ml_df['label_raw'].apply(self.infer_label_from_text)
        
        # Extract numerical features if available
        if field_mapping['experience_fields']:
            exp_col = field_mapping['experience_fields'][0]
            ml_df['years_experience'] = pd.to_numeric(df[exp_col], errors='coerce').fillna(3)
        else:
            # Generate synthetic experience data
            ml_df['years_experience'] = np.random.randint(1, 10, len(ml_df))
        
        # Extract education if available
        if field_mapping['education_fields']:
            edu_col = field_mapping['education_fields'][0]
            ml_df['education'] = df[edu_col].fillna('Bachelor')
        else:
            # Generate synthetic education data
            ml_df['education'] = np.random.choice(['Bachelor', 'Master', 'PhD'], len(ml_df))
        
        # Calculate derived features
        if 'resume_text' in ml_df.columns and 'job_description' in ml_df.columns:
            ml_df['keyword_overlap_score'] = ml_df.apply(self.calculate_keyword_overlap, axis=1)
        
        # Remove rows with missing critical data
        if 'is_match' in ml_df.columns:
            ml_df = ml_df.dropna(subset=['is_match'])
        
        print(f"✅ Prepared dataset shape: {ml_df.shape}")
        if 'is_match' in ml_df.columns:
            print(f"📊 Label distribution: {ml_df['is_match'].value_counts().to_dict()}")
        
        return ml_df
    
    def infer_label_from_text(self, label_text: str) -> int:
        """Infer binary label from text description"""
        if pd.isna(label_text):
            return 0
        
        label_str = str(label_text).lower()
        
        # Check for positive indicators
        if any(word in label_str for word in ['match', 'fit', 'suitable', 'good', 'yes', '1', 'true']):
            return 1
        
        # Check for negative indicators  
        if any(word in label_str for word in ['mismatch', 'no fit', 'unsuitable', 'bad', 'no', '0', 'false']):
            return 0
        
        # Default to no match if unclear
        return 0
    
    def calculate_keyword_overlap(self, row) -> float:
        """Calculate keyword overlap between job description and resume"""
        try:
            job_text = str(row.get('job_description', '')).lower()
            resume_text = str(row.get('resume_text', '')).lower()
            
            job_keywords = set(re.findall(r'\b\w+\b', job_text))
            resume_keywords = set(re.findall(r'\b\w+\b', resume_text))
            
            if len(job_keywords) == 0:
                return 0.0
            
            overlap = len(job_keywords.intersection(resume_keywords))
            return overlap / len(job_keywords)
            
        except Exception:
            return 0.0
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str = "processed_resume_dataset.csv"):
        """Save the processed dataset to CSV"""
        try:
            df.to_csv(output_path, index=False)
            print(f"💾 Processed dataset saved to: {output_path}")
            print(f"📊 Saved {len(df)} rows with {len(df.columns)} columns")
            
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")

def demonstrate_ml_training(df: pd.DataFrame):
    """Demonstrate ML training with the loaded dataset"""
    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn not available for ML training demo")
        return
    
    if 'is_match' not in df.columns:
        print("❌ No target variable found for ML training")
        return
    
    print("\n🎓 ML Training Demonstration")
    print("=" * 50)
    
    try:
        # Prepare features
        features_df = pd.DataFrame()
        
        # Add numerical features
        numerical_cols = ['years_experience', 'keyword_overlap_score']
        for col in numerical_cols:
            if col in df.columns:
                features_df[col] = df[col].fillna(df[col].mean())
        
        # Add categorical features
        if 'education' in df.columns:
            le = LabelEncoder()
            features_df['education_encoded'] = le.fit_transform(df['education'].fillna('Bachelor'))
        
        # Add text features (simplified TF-IDF)
        if 'resume_text' in df.columns and 'job_description' in df.columns:
            combined_text = df['resume_text'].fillna('') + ' ' + df['job_description'].fillna('')
            
            # Use a small TF-IDF for demo
            tfidf = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_features = tfidf.fit_transform(combined_text)
            
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                                   columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
            
            features_df = pd.concat([features_df.reset_index(drop=True), 
                                   tfidf_df.reset_index(drop=True)], axis=1)
        
        # Prepare target
        y = df['is_match'].values
        X = features_df.values
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            print("❌ No features available for training")
            return
        
        print(f"📊 Training with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n📈 Model Performance:")
        print(f"   Accuracy:  {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f}")
        print(f"   F1-Score:  {f1:.3f}")
        
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Match', 'Match']))
        
    except Exception as e:
        print(f"❌ Error in ML training: {e}")

def main():
    """Main function"""
    print("🤖 Hugging Face Resume Dataset Loader")
    print("=" * 60)
    
    if not DATASETS_AVAILABLE:
        print("❌ datasets library not available")
        print("💡 Install with: pip install datasets")
        return
    
    try:
        # Initialize loader
        loader = HuggingFaceResumeDatasetLoader()
        
        # Load dataset
        df = loader.load_dataset("netsol/resume-score-details")
        
        # Inspect dataset
        field_mapping = loader.inspect_dataset(df)
        
        # Prepare for ML
        ml_df = loader.prepare_ml_data(df, field_mapping)
        
        # Save processed data
        loader.save_processed_data(ml_df)
        
        # Demonstrate ML training
        if len(ml_df) > 0:
            demonstrate_ml_training(ml_df)
        
        print(f"\n🎉 Dataset processing completed!")
        print(f"📊 Final dataset: {len(ml_df)} samples ready for ML")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Provide helpful guidance
        if "ConnectionError" in str(e) or "Couldn't reach" in str(e):
            print("\n💡 Troubleshooting network issues:")
            print("   1. Check your internet connection")
            print("   2. Try again in a few minutes")
            print("   3. Check if the dataset still exists on Hugging Face")
            print("   4. Consider using the local demo script instead:")
            print("      python src/scripts/ml_demo_local.py")

if __name__ == "__main__":
    main()
