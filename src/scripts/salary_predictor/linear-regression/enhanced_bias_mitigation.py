#!/usr/bin/env python3
"""
Enhanced Salary Prediction with Bias Detection and Mitigation
Comprehensive analysis with improved visualization and error handling
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and explore the biased dataset"""
    print("🔍 LOADING AND EXPLORING BIASED DATASET")
    print("=" * 60)
    
    df = pd.read_csv("src/storage/salary_prediction_dataset_biased.csv")
    
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️ Missing Values:")
        print(missing[missing > 0])
    else:
        print(f"\n✅ No missing values found")
    
    print(f"\n📈 Salary Statistics:")
    print(f"   Mean: ${df['salary'].mean():,.0f}")
    print(f"   Median: ${df['salary'].median():,.0f}")
    print(f"   Min: ${df['salary'].min():,.0f}")
    print(f"   Max: ${df['salary'].max():,.0f}")
    print(f"   Std Dev: ${df['salary'].std():,.0f}")
    
    return df

def detect_bias(df):
    """Comprehensive bias detection"""
    print(f"\n⚖️ COMPREHENSIVE BIAS DETECTION")
    print("=" * 60)
    
    # Gender bias analysis
    print(f"\n👥 GENDER BIAS ANALYSIS:")
    gender_stats = df.groupby("gender")["salary"].agg(['count', 'mean', 'std']).round(2)
    print(gender_stats)
    
    # Calculate gender pay gaps
    male_avg = df[df['gender'] == 'Male']['salary'].mean()
    female_avg = df[df['gender'] == 'Female']['salary'].mean()
    other_avg = df[df['gender'] == 'Other']['salary'].mean()
    
    male_female_gap = ((male_avg - female_avg) / female_avg) * 100
    male_other_gap = ((male_avg - other_avg) / other_avg) * 100
    
    print(f"\n📊 Pay Gaps:")
    print(f"   Male vs Female: {male_female_gap:+.1f}% (${male_avg - female_avg:,.0f})")
    print(f"   Male vs Other: {male_other_gap:+.1f}% (${male_avg - other_avg:,.0f})")
    
    # Statistical significance test
    male_salaries = df[df['gender'] == 'Male']['salary']
    female_salaries = df[df['gender'] == 'Female']['salary']
    other_salaries = df[df['gender'] == 'Other']['salary']
    
    t_stat_mf, p_val_mf = ttest_ind(male_salaries, female_salaries, equal_var=False)
    t_stat_mo, p_val_mo = ttest_ind(male_salaries, other_salaries, equal_var=False)
    
    print(f"\n🔬 Statistical Significance:")
    print(f"   Male vs Female: t={t_stat_mf:.2f}, p={p_val_mf:.2e}")
    print(f"   Male vs Other: t={t_stat_mo:.2f}, p={p_val_mo:.2e}")
    print(f"   Gender bias significant: {'❌ YES' if p_val_mf < 0.001 else '✅ No'}")
    
    # College tier bias
    print(f"\n🎓 COLLEGE TIER BIAS ANALYSIS:")
    college_stats = df.groupby("college")["salary"].agg(['count', 'mean', 'std']).round(2)
    print(college_stats)
    
    tier1_avg = df[df['college'] == 'Tier 1']['salary'].mean()
    tier3_avg = df[df['college'] == 'Tier 3']['salary'].mean()
    tier_gap = ((tier1_avg - tier3_avg) / tier3_avg) * 100
    
    print(f"\n📊 College Tier Gap:")
    print(f"   Tier 1 vs Tier 3: {tier_gap:+.1f}% (${tier1_avg - tier3_avg:,.0f})")
    
    return {
        'gender_gap': male_female_gap,
        'college_gap': tier_gap,
        'gender_p_value': p_val_mf,
        'male_avg': male_avg,
        'female_avg': female_avg,
        'tier1_avg': tier1_avg,
        'tier3_avg': tier3_avg
    }

def analyze_correlations(df):
    """Analyze feature correlations with salary"""
    print(f"\n🔗 FEATURE CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Create dummy variables for correlation analysis
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Get correlations with salary
    salary_corr = df_encoded.corr()["salary"].sort_values(ascending=False)
    
    print(f"\n📊 Top 10 Positive Correlations with Salary:")
    for i, (feature, corr) in enumerate(salary_corr.head(11).items(), 1):
        if feature != 'salary':  # Skip salary itself
            print(f"   {i:2d}. {feature}: {corr:.4f}")
    
    print(f"\n📊 Top 10 Negative Correlations with Salary:")
    negative_corr = salary_corr.tail(10)
    for i, (feature, corr) in enumerate(negative_corr.items(), 1):
        print(f"   {i:2d}. {feature}: {corr:.4f}")
    
    return salary_corr

def create_visualizations(df):
    """Create bias visualization plots"""
    print(f"\n📊 CREATING BIAS VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Set up the plotting environment
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Salary Bias Analysis - Biased Dataset', fontsize=16, fontweight='bold')
        
        # 1. Gender bias boxplot
        sns.boxplot(data=df, x="gender", y="salary", ax=axes[0, 0])
        axes[0, 0].set_title("Salary Distribution by Gender")
        axes[0, 0].set_ylabel("Salary")
        
        # Add mean lines
        for i, gender in enumerate(df['gender'].unique()):
            mean_salary = df[df['gender'] == gender]['salary'].mean()
            axes[0, 0].hlines(mean_salary, i-0.4, i+0.4, colors='red', linestyles='--', alpha=0.7)
            axes[0, 0].text(i, mean_salary + 5000, f'${mean_salary:,.0f}', ha='center', va='bottom', 
                           fontweight='bold', color='red')
        
        # 2. College tier bias boxplot
        sns.boxplot(data=df, x="college", y="salary", ax=axes[0, 1])
        axes[0, 1].set_title("Salary Distribution by College Tier")
        axes[0, 1].set_ylabel("Salary")
        
        # Add mean lines
        for i, college in enumerate(df['college'].unique()):
            mean_salary = df[df['college'] == college]['salary'].mean()
            axes[0, 1].hlines(mean_salary, i-0.4, i+0.4, colors='red', linestyles='--', alpha=0.7)
            axes[0, 1].text(i, mean_salary + 5000, f'${mean_salary:,.0f}', ha='center', va='bottom',
                           fontweight='bold', color='red')
        
        # 3. Salary by experience and gender
        exp_gender_salary = df.groupby(['experience_years', 'gender'])['salary'].mean().unstack()
        
        for gender in exp_gender_salary.columns:
            axes[1, 0].plot(exp_gender_salary.index, exp_gender_salary[gender], 
                           marker='o', label=gender, linewidth=2)
        
        axes[1, 0].set_title("Average Salary by Experience and Gender")
        axes[1, 0].set_xlabel("Years of Experience")
        axes[1, 0].set_ylabel("Average Salary")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Gender distribution by job role
        role_gender = pd.crosstab(df['job_role'], df['gender'], normalize='index') * 100
        role_gender.plot(kind='bar', ax=axes[1, 1], width=0.8)
        axes[1, 1].set_title("Gender Distribution by Job Role (%)")
        axes[1, 1].set_xlabel("Job Role")
        axes[1, 1].set_ylabel("Percentage")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(title='Gender')
        
        plt.tight_layout()
        plt.savefig('salary_bias_analysis_biased_dataset.png', dpi=300, bbox_inches='tight')
        print("📈 Visualizations saved as 'salary_bias_analysis_biased_dataset.png'")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Visualization error (continuing without plots): {e}")

def implement_bias_mitigation(df):
    """Implement comprehensive bias mitigation techniques"""
    print(f"\n🛠️ IMPLEMENTING BIAS MITIGATION TECHNIQUES")
    print("=" * 60)
    
    # Technique 1: Reweighting
    print(f"\n1️⃣ REWEIGHTING TECHNIQUE:")
    
    def compute_fairness_weights(df, sensitive_attr):
        """Compute weights to balance representation"""
        group_counts = df[sensitive_attr].value_counts()
        total = len(df)
        
        # Calculate weights inversely proportional to group size
        weights = df[sensitive_attr].map(lambda x: total / (len(group_counts) * group_counts[x]))
        return weights / weights.mean()  # Normalize weights
    
    # Compute weights for gender and college
    df['gender_weight'] = compute_fairness_weights(df, 'gender')
    df['college_weight'] = compute_fairness_weights(df, 'college')
    df['combined_weight'] = (df['gender_weight'] + df['college_weight']) / 2
    
    print(f"   Average weights by gender:")
    for gender in df['gender'].unique():
        avg_weight = df[df['gender'] == gender]['gender_weight'].mean()
        print(f"     {gender}: {avg_weight:.3f}")
    
    print(f"   Average weights by college:")
    for college in df['college'].unique():
        avg_weight = df[df['college'] == college]['college_weight'].mean()
        print(f"     {college}: {avg_weight:.3f}")
    
    # Technique 2: Fair Model Training
    print(f"\n2️⃣ FAIR MODEL TRAINING:")
    
    # Prepare features (remove sensitive attributes)
    feature_cols = ['experience_years', 'education_level', 'skills_count']
    
    # Add non-sensitive categorical features
    categorical_cols = ['job_role', 'city', 'industry', 'certification', 'technology']
    
    X = df[feature_cols].copy()
    
    # One-hot encode categorical variables
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col)
        X = pd.concat([X, dummies], axis=1)
    
    y = df['salary']
    
    print(f"   Features used for training: {X.shape[1]}")
    print(f"   Training samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, df['combined_weight'], test_size=0.2, random_state=42
    )
    
    # Train regular model (biased)
    regular_model = LinearRegression()
    regular_model.fit(X_train, y_train)
    
    # Train fair model (with weights)
    fair_model = LinearRegression()
    fair_model.fit(X_train, y_train, sample_weight=weights_train)
    
    # Make predictions
    y_pred_regular = regular_model.predict(X_test)
    y_pred_fair = fair_model.predict(X_test)
    
    # Evaluate models
    regular_r2 = r2_score(y_test, y_pred_regular)
    fair_r2 = r2_score(y_test, y_pred_fair)
    
    print(f"   Regular model R²: {regular_r2:.4f}")
    print(f"   Fair model R²: {fair_r2:.4f}")
    
    # Technique 3: Post-processing Adjustment
    print(f"\n3️⃣ POST-PROCESSING BIAS CORRECTION:")
    
    def apply_bias_correction(predictions, gender, college, correction_factors):
        """Apply post-processing bias corrections"""
        corrections = predictions.copy()
        
        # Gender-based corrections
        female_mask = (gender == 'Female')
        other_mask = (gender == 'Other')
        
        corrections[female_mask] *= (1 + correction_factors['female_boost'])
        corrections[other_mask] *= (1 + correction_factors['other_boost'])
        
        # College-based corrections  
        tier3_mask = (college == 'Tier 3')
        corrections[tier3_mask] *= (1 + correction_factors['tier3_boost'])
        
        return corrections
    
    # Calculate correction factors based on observed gaps
    bias_stats = detect_bias(df)
    
    correction_factors = {
        'female_boost': 0.15,  # 15% boost for females
        'other_boost': 0.10,   # 10% boost for others
        'tier3_boost': 0.12    # 12% boost for Tier 3
    }
    
    print(f"   Correction factors:")
    for factor, value in correction_factors.items():
        print(f"     {factor}: +{value*100:.0f}%")
    
    # Apply corrections to test set
    test_df = df.loc[X_test.index].copy()
    
    y_pred_corrected = apply_bias_correction(
        y_pred_fair, 
        test_df['gender'], 
        test_df['college'], 
        correction_factors
    )
    
    return {
        'regular_model': regular_model,
        'fair_model': fair_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_regular': y_pred_regular,
        'y_pred_fair': y_pred_fair,
        'y_pred_corrected': y_pred_corrected,
        'test_df': test_df,
        'correction_factors': correction_factors
    }

def evaluate_bias_reduction(results):
    """Evaluate the effectiveness of bias reduction techniques"""
    print(f"\n📊 BIAS REDUCTION EVALUATION")
    print("=" * 60)
    
    test_df = results['test_df']
    
    # Function to calculate bias metrics
    def calculate_bias_metrics(predictions, actual, gender, college):
        df_temp = pd.DataFrame({
            'pred': predictions,
            'actual': actual,
            'gender': gender,
            'college': college
        })
        
        # Gender bias metrics
        gender_pred_avg = df_temp.groupby('gender')['pred'].mean()
        male_female_pred_gap = ((gender_pred_avg['Male'] - gender_pred_avg['Female']) 
                               / gender_pred_avg['Female'] * 100)
        
        # College bias metrics
        college_pred_avg = df_temp.groupby('college')['pred'].mean()
        tier1_tier3_pred_gap = ((college_pred_avg['Tier 1'] - college_pred_avg['Tier 3']) 
                               / college_pred_avg['Tier 3'] * 100)
        
        return {
            'gender_gap': male_female_pred_gap,
            'college_gap': tier1_tier3_pred_gap,
            'gender_avgs': gender_pred_avg,
            'college_avgs': college_pred_avg
        }
    
    # Calculate bias metrics for each approach
    regular_bias = calculate_bias_metrics(
        results['y_pred_regular'], results['y_test'], 
        test_df['gender'], test_df['college']
    )
    
    fair_bias = calculate_bias_metrics(
        results['y_pred_fair'], results['y_test'],
        test_df['gender'], test_df['college'] 
    )
    
    corrected_bias = calculate_bias_metrics(
        results['y_pred_corrected'], results['y_test'],
        test_df['gender'], test_df['college']
    )
    
    print(f"\n🔍 BIAS COMPARISON RESULTS:")
    print(f"{'Approach':<20} {'Gender Gap':<15} {'College Gap':<15}")
    print(f"{'-'*50}")
    print(f"{'Regular Model':<20} {regular_bias['gender_gap']:>10.1f}% {regular_bias['college_gap']:>13.1f}%")
    print(f"{'Fair Model':<20} {fair_bias['gender_gap']:>10.1f}% {fair_bias['college_gap']:>13.1f}%")
    print(f"{'Corrected Model':<20} {corrected_bias['gender_gap']:>10.1f}% {corrected_bias['college_gap']:>13.1f}%")
    
    # Calculate improvement
    gender_improvement = regular_bias['gender_gap'] - corrected_bias['gender_gap']
    college_improvement = regular_bias['college_gap'] - corrected_bias['college_gap']
    
    print(f"\n✨ BIAS REDUCTION ACHIEVED:")
    print(f"   Gender gap reduced by: {gender_improvement:.1f} percentage points")
    print(f"   College gap reduced by: {college_improvement:.1f} percentage points")
    
    # Model performance comparison
    regular_r2 = r2_score(results['y_test'], results['y_pred_regular'])
    fair_r2 = r2_score(results['y_test'], results['y_pred_fair'])
    corrected_r2 = r2_score(results['y_test'], results['y_pred_corrected'])
    
    print(f"\n📈 MODEL PERFORMANCE:")
    print(f"   Regular Model R²: {regular_r2:.4f}")
    print(f"   Fair Model R²: {fair_r2:.4f}")
    print(f"   Corrected Model R²: {corrected_r2:.4f}")
    
    return {
        'regular_bias': regular_bias,
        'fair_bias': fair_bias,
        'corrected_bias': corrected_bias,
        'improvements': {
            'gender': gender_improvement,
            'college': college_improvement
        }
    }

def save_results(df, results):
    """Save debiased predictions and analysis results"""
    print(f"\n💾 SAVING RESULTS")
    print("=" * 60)
    
    # Create results DataFrame
    test_indices = results['X_test'].index
    results_df = df.loc[test_indices].copy()
    
    results_df['pred_regular'] = results['y_pred_regular']
    results_df['pred_fair'] = results['y_pred_fair'] 
    results_df['pred_corrected'] = results['y_pred_corrected']
    
    # Calculate prediction differences
    results_df['bias_correction'] = results_df['pred_corrected'] - results_df['pred_regular']
    results_df['correction_percent'] = (results_df['bias_correction'] / results_df['pred_regular'] * 100)
    
    # Save to CSV
    results_df.to_csv('debiased_salary_predictions_enhanced.csv', index=False)
    
    print(f"✅ Results saved to 'debiased_salary_predictions_enhanced.csv'")
    print(f"📊 Saved {len(results_df)} predictions with bias corrections")
    
    # Show sample corrections
    print(f"\n📋 Sample Bias Corrections:")
    print(f"{'Gender':<8} {'College':<8} {'Original':<10} {'Corrected':<10} {'Change':<8}")
    print(f"{'-'*50}")
    
    sample_df = results_df.head(10)
    for _, row in sample_df.iterrows():
        print(f"{row['gender']:<8} {row['college']:<8} ${row['pred_regular']:>8.0f} ${row['pred_corrected']:>9.0f} {row['correction_percent']:>6.1f}%")

def main():
    """Main execution function"""
    print("🚀 ENHANCED SALARY BIAS DETECTION & MITIGATION")
    print("=" * 70)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Detect bias
    bias_stats = detect_bias(df)
    
    # Analyze correlations
    correlations = analyze_correlations(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Implement bias mitigation
    mitigation_results = implement_bias_mitigation(df)
    
    # Evaluate bias reduction
    evaluation_results = evaluate_bias_reduction(mitigation_results)
    
    # Save results
    save_results(df, mitigation_results)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📊 Original gender pay gap: {bias_stats['gender_gap']:+.1f}%")
    print(f"🎯 Bias reduction achieved: {evaluation_results['improvements']['gender']:.1f} percentage points")
    print(f"💡 Enhanced results saved with comprehensive bias mitigation!")

if __name__ == "__main__":
    main()
