#!/usr/bin/env python3
"""
Quick Bias Comparison Script
Compare bias levels between different salary datasets
"""

import pandas as pd
import numpy as np
from scipy import stats

def quick_bias_check(csv_file, dataset_name):
    """Quick bias analysis for any salary dataset"""
    try:
        df = pd.read_csv(csv_file)
        print(f"\n{'='*50}")
        print(f"📊 {dataset_name}")
        print(f"{'='*50}")
        
        # Basic stats
        total = len(df)
        print(f"Records: {total:,}")
        
        # Gender distribution
        gender_counts = df['gender'].value_counts()
        print(f"\n👥 Gender Distribution:")
        for gender, count in gender_counts.items():
            pct = (count/total) * 100
            print(f"   {gender}: {count:,} ({pct:.2f}%)")
        
        # Salary statistics by gender
        salary_stats = df.groupby('gender')['salary'].mean()
        print(f"\n💰 Average Salaries:")
        for gender, avg_salary in salary_stats.items():
            print(f"   {gender}: ${avg_salary:,.0f}")
        
        # Pay gaps
        male_avg = salary_stats.get('Male', 0)
        female_avg = salary_stats.get('Female', 0)
        other_avg = salary_stats.get('Other', 0)
        
        if male_avg > 0 and female_avg > 0:
            male_female_gap = ((male_avg - female_avg) / female_avg) * 100
            print(f"\n📉 Pay Gaps:")
            print(f"   Male vs Female: {male_female_gap:+.3f}%")
            
            if other_avg > 0:
                male_other_gap = ((male_avg - other_avg) / other_avg) * 100
                female_other_gap = ((female_avg - other_avg) / other_avg) * 100
                print(f"   Male vs Other: {male_other_gap:+.3f}%")
                print(f"   Female vs Other: {female_other_gap:+.3f}%")
        
        # Statistical test
        male_salaries = df[df['gender'] == 'Male']['salary']
        female_salaries = df[df['gender'] == 'Female']['salary']
        other_salaries = df[df['gender'] == 'Other']['salary']
        
        if len(male_salaries) > 0 and len(female_salaries) > 0:
            if len(other_salaries) > 0:
                f_stat, p_value = stats.f_oneway(male_salaries, female_salaries, other_salaries)
                test_name = "ANOVA"
            else:
                f_stat, p_value = stats.ttest_ind(male_salaries, female_salaries)
                test_name = "T-test"
            
            print(f"\n🔬 Statistical Test ({test_name}):")
            print(f"   P-value: {p_value:.2e}")
            print(f"   Significant bias: {'❌ Yes' if p_value < 0.05 else '✅ No'}")
        
        # Quick bias score
        max_gap = max(abs(male_female_gap) if male_avg > 0 and female_avg > 0 else 0,
                     abs(male_other_gap) if male_avg > 0 and other_avg > 0 else 0,
                     abs(female_other_gap) if female_avg > 0 and other_avg > 0 else 0)
        
        # Gender representation balance
        percentages = [count/total*100 for count in gender_counts.values]
        representation_imbalance = max(percentages) - min(percentages)
        
        # Simple bias score calculation
        gap_score = min(max_gap * 2, 50)  # Up to 50 points for pay gap
        rep_score = min(representation_imbalance * 1.5, 30)  # Up to 30 points for representation
        sig_score = 20 if p_value < 0.05 else 0  # 20 points if statistically significant
        
        total_bias_score = gap_score + rep_score + sig_score
        
        print(f"\n🎯 Quick Bias Score: {total_bias_score:.1f}/100")
        
        if total_bias_score < 10:
            level = "🟢 VERY LOW"
        elif total_bias_score < 25:
            level = "🟡 LOW"
        elif total_bias_score < 50:
            level = "🟠 MODERATE"
        else:
            level = "🔴 HIGH"
        
        print(f"   Bias Level: {level}")
        
        return {
            'dataset': dataset_name,
            'total_records': total,
            'max_pay_gap': max_gap,
            'representation_imbalance': representation_imbalance,
            'p_value': p_value,
            'bias_score': total_bias_score,
            'bias_level': level
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")
        return None

def compare_datasets():
    """Compare multiple salary datasets"""
    print("🔍 SALARY DATASET BIAS COMPARISON")
    print("=" * 70)
    
    datasets = [
        ('src/storage/salary_prediction_dataset_100k_without_bias.csv', 'Without Bias Dataset'),
        ('src/storage/salary_prediction_dataset_100k_with_gender.csv', 'With Gender Dataset'),
    ]
    
    # Try to load biased dataset if it exists
    try:
        pd.read_csv('src/storage/salary_prediction_dataset_biased.csv')
        datasets.append(('src/storage/salary_prediction_dataset_biased.csv', 'Biased Dataset'))
    except:
        pass
    
    results = []
    
    for csv_file, name in datasets:
        result = quick_bias_check(csv_file, name)
        if result:
            results.append(result)
    
    # Comparison summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"📋 COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n🎯 Bias Scores Comparison:")
        sorted_results = sorted(results, key=lambda x: x['bias_score'])
        
        for i, result in enumerate(sorted_results):
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            print(f"   {rank_emoji} {result['dataset']}: {result['bias_score']:.1f}/100 {result['bias_level']}")
        
        print(f"\n📊 Key Metrics:")
        print(f"{'Dataset':<25} {'Pay Gap':<12} {'Rep. Imbal.':<12} {'P-value':<12}")
        print(f"{'-'*65}")
        
        for result in results:
            print(f"{result['dataset'][:24]:<25} {result['max_pay_gap']:>10.3f}% {result['representation_imbalance']:>10.2f}% {result['p_value']:>10.2e}")
        
        # Best and worst
        best = min(results, key=lambda x: x['bias_score'])
        worst = max(results, key=lambda x: x['bias_score'])
        
        print(f"\n🏆 BEST (Least Biased): {best['dataset']}")
        print(f"   Score: {best['bias_score']:.1f}/100")
        print(f"   Max Pay Gap: {best['max_pay_gap']:.3f}%")
        
        if len(results) > 1:
            print(f"\n⚠️ WORST (Most Biased): {worst['dataset']}")
            print(f"   Score: {worst['bias_score']:.1f}/100") 
            print(f"   Max Pay Gap: {worst['max_pay_gap']:.3f}%")
        
        improvement = worst['bias_score'] - best['bias_score']
        if improvement > 10:
            print(f"\n✨ IMPROVEMENT: {improvement:.1f} point reduction in bias score!")

if __name__ == "__main__":
    compare_datasets()
    print(f"\n🎉 Bias comparison completed!")
