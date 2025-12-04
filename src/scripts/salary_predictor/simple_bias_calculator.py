#!/usr/bin/env python3
"""
Simple Salary Bias Calculator
Quick bias analysis for salary datasets
"""

import pandas as pd
import numpy as np

def simple_bias_analysis():
    """Simple and focused bias analysis"""
    print("🔍 SIMPLE SALARY BIAS ANALYSIS")
    print("=" * 50)
    
    # Load data
    try:
        df = pd.read_csv('src/storage/salary_prediction_dataset_100k_with_gender.csv')
        print(f"✅ Loaded {len(df):,} salary records")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 1. Basic Stats
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Dataset size: {len(df):,} records")
    print(f"   Salary range: ${df['salary'].min():,} - ${df['salary'].max():,}")
    print(f"   Average salary: ${df['salary'].mean():,.0f}")
    
    # 2. Gender Distribution
    print(f"\n👥 GENDER BREAKDOWN:")
    gender_counts = df['gender'].value_counts()
    total = len(df)
    for gender, count in gender_counts.items():
        pct = (count/total) * 100
        print(f"   {gender}: {count:,} ({pct:.1f}%)")
    
    # 3. Salary by Gender
    print(f"\n💰 SALARY BY GENDER:")
    salary_by_gender = df.groupby('gender')['salary'].agg(['count', 'mean', 'median'])
    for gender in salary_by_gender.index:
        avg_salary = salary_by_gender.loc[gender, 'mean']
        median_salary = salary_by_gender.loc[gender, 'median']
        print(f"   {gender}: Avg ${avg_salary:,.0f}, Median ${median_salary:,.0f}")
    
    # 4. Pay Gap Analysis
    print(f"\n📉 PAY GAP ANALYSIS:")
    male_avg = df[df['gender'] == 'Male']['salary'].mean()
    female_avg = df[df['gender'] == 'Female']['salary'].mean()
    other_avg = df[df['gender'] == 'Other']['salary'].mean()
    
    # Calculate gaps
    male_female_gap = ((male_avg - female_avg) / female_avg) * 100
    male_other_gap = ((male_avg - other_avg) / other_avg) * 100
    
    print(f"   Male vs Female: {male_female_gap:+.2f}%")
    print(f"   Male vs Other: {male_other_gap:+.2f}%")
    
    # 5. Job Role Analysis (Top roles with biggest gaps)
    print(f"\n👔 JOB ROLE BIAS (Top 5 roles with biggest gaps):")
    
    job_gaps = []
    for role in df['job_role'].unique():
        role_data = df[df['job_role'] == role]
        male_data = role_data[role_data['gender'] == 'Male']['salary']
        female_data = role_data[role_data['gender'] == 'Female']['salary']
        
        if len(male_data) > 50 and len(female_data) > 50:  # Only roles with enough data
            gap = ((male_data.mean() - female_data.mean()) / female_data.mean()) * 100
            job_gaps.append({
                'role': role,
                'gap': gap,
                'male_avg': male_data.mean(),
                'female_avg': female_data.mean()
            })
    
    # Sort by absolute gap size
    job_gaps_df = pd.DataFrame(job_gaps)
    job_gaps_df = job_gaps_df.reindex(job_gaps_df['gap'].abs().sort_values(ascending=False).index)
    
    for _, row in job_gaps_df.head().iterrows():
        status = "📈" if row['gap'] > 0 else "📉"
        print(f"   {status} {row['role']}: {row['gap']:+.1f}% gap")
    
    # 6. Bias Summary
    print(f"\n⚖️ BIAS SUMMARY:")
    
    # Overall assessment
    overall_gap = abs(male_female_gap)
    if overall_gap < 2:
        bias_level = "🟢 LOW"
    elif overall_gap < 5:
        bias_level = "🟡 MODERATE" 
    else:
        bias_level = "🔴 HIGH"
    
    print(f"   Overall Gender Bias Level: {bias_level}")
    print(f"   Main Pay Gap: {male_female_gap:+.2f}%")
    
    # Check for representation bias
    gender_percentages = [(count/total*100) for count in gender_counts.values]
    min_gender_pct = min(gender_percentages)
    if min_gender_pct < 20:
        print(f"   🔴 Representation Issue: Minority gender only {min_gender_pct:.1f}%")
    else:
        print(f"   🟢 Good Gender Representation: {min_gender_pct:.1f}% minimum")
    
    # Quick recommendations
    print(f"\n💡 QUICK RECOMMENDATIONS:")
    if overall_gap > 5:
        print(f"   • Immediate salary equity review needed")
        print(f"   • Review hiring and promotion practices")
    elif overall_gap > 2:
        print(f"   • Monitor salary decisions for bias")
        print(f"   • Consider structured pay scales")
    else:
        print(f"   • Continue regular monitoring")
    
    print(f"   • Implement bias training for managers")
    print(f"   • Use structured interviews and evaluation criteria")
    
    return df

def calculate_bias_score(df):
    """Calculate a simple bias score (0-100, where 0 = no bias, 100 = extreme bias)"""
    
    # Gender pay gap component (0-40 points)
    male_avg = df[df['gender'] == 'Male']['salary'].mean()
    female_avg = df[df['gender'] == 'Female']['salary'].mean()
    pay_gap = abs((male_avg - female_avg) / female_avg * 100)
    gap_score = min(pay_gap * 4, 40)  # Cap at 40 points
    
    # Representation component (0-30 points)
    gender_counts = df['gender'].value_counts()
    total = len(df)
    percentages = [count/total*100 for count in gender_counts.values()]
    ideal_pct = 100 / len(percentages)  # Ideal would be equal representation
    
    representation_bias = sum([abs(pct - ideal_pct) for pct in percentages]) / len(percentages)
    rep_score = min(representation_bias * 2, 30)  # Cap at 30 points
    
    # Role distribution bias (0-30 points)
    role_bias_scores = []
    for role in df['job_role'].unique():
        role_data = df[df['job_role'] == role]
        role_gender_dist = role_data['gender'].value_counts(normalize=True) * 100
        if len(role_gender_dist) > 1:
            role_bias = max(role_gender_dist) - min(role_gender_dist)
            role_bias_scores.append(role_bias)
    
    avg_role_bias = np.mean(role_bias_scores) if role_bias_scores else 0
    role_score = min(avg_role_bias * 0.5, 30)  # Cap at 30 points
    
    total_bias_score = gap_score + rep_score + role_score
    
    print(f"\n🎯 BIAS SCORE BREAKDOWN:")
    print(f"   Pay Gap Score: {gap_score:.1f}/40 (Gap: {pay_gap:.1f}%)")
    print(f"   Representation Score: {rep_score:.1f}/30")
    print(f"   Role Distribution Score: {role_score:.1f}/30")
    print(f"   TOTAL BIAS SCORE: {total_bias_score:.1f}/100")
    
    if total_bias_score < 20:
        level = "🟢 LOW BIAS"
    elif total_bias_score < 50:
        level = "🟡 MODERATE BIAS"
    else:
        level = "🔴 HIGH BIAS"
    
    print(f"   Assessment: {level}")
    
    return total_bias_score

if __name__ == "__main__":
    df = simple_bias_analysis()
    if df is not None:
        calculate_bias_score(df)
        print(f"\n🎉 Analysis Complete!")
