#!/usr/bin/env python3
"""
Bias Analysis for "Without Bias" Salary Dataset
Comprehensive analysis to verify if dataset truly has reduced bias
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_unbiased_dataset():
    """Load the 'without bias' salary dataset"""
    try:
        print("📥 Loading 'without bias' salary dataset...")
        df = pd.read_csv('src/storage/salary_prediction_dataset_100k_without_bias.csv')
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def compare_with_biased_dataset():
    """Compare with the regular biased dataset if available"""
    try:
        df_biased = pd.read_csv('src/storage/salary_prediction_dataset_100k_with_gender.csv')
        return df_biased
    except:
        print("⚠️ Biased dataset not found for comparison")
        return None

def analyze_gender_equity(df):
    """Comprehensive gender equity analysis"""
    print("\n" + "="*60)
    print("⚖️ GENDER EQUITY ANALYSIS")
    print("="*60)
    
    # Basic gender distribution
    gender_dist = df['gender'].value_counts()
    total = len(df)
    
    print(f"\n👥 Gender Distribution:")
    for gender, count in gender_dist.items():
        pct = (count / total) * 100
        print(f"   {gender}: {count:,} ({pct:.2f}%)")
    
    # Test for equal representation
    expected_per_gender = total / len(gender_dist)
    chi2_stat, p_value = stats.chisquare(gender_dist.values)
    
    print(f"\n🔬 Gender Representation Test:")
    print(f"   Expected per gender: {expected_per_gender:,.0f}")
    print(f"   Chi-square statistic: {chi2_stat:.4f}")
    print(f"   P-value: {p_value:.4e}")
    print(f"   Equal representation: {'✅ Yes' if p_value > 0.05 else '❌ No'}")
    
    # Salary equity analysis
    salary_stats = df.groupby('gender')['salary'].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)
    
    print(f"\n💰 Salary Statistics by Gender:")
    print(salary_stats)
    
    # Calculate pay gaps
    male_avg = df[df['gender'] == 'Male']['salary'].mean()
    female_avg = df[df['gender'] == 'Female']['salary'].mean()
    other_avg = df[df['gender'] == 'Other']['salary'].mean()
    
    male_female_gap = ((male_avg - female_avg) / female_avg) * 100
    male_other_gap = ((male_avg - other_avg) / other_avg) * 100
    female_other_gap = ((female_avg - other_avg) / other_avg) * 100
    
    print(f"\n📊 Pay Gap Analysis:")
    print(f"   Male average: ${male_avg:,.2f}")
    print(f"   Female average: ${female_avg:,.2f}")
    print(f"   Other average: ${other_avg:,.2f}")
    print(f"   Male-Female gap: {male_female_gap:+.3f}%")
    print(f"   Male-Other gap: {male_other_gap:+.3f}%")
    print(f"   Female-Other gap: {female_other_gap:+.3f}%")
    
    # Statistical significance tests
    male_salaries = df[df['gender'] == 'Male']['salary']
    female_salaries = df[df['gender'] == 'Female']['salary']
    other_salaries = df[df['gender'] == 'Other']['salary']
    
    # ANOVA test for all groups
    f_stat, p_anova = stats.f_oneway(male_salaries, female_salaries, other_salaries)
    
    print(f"\n🔬 Statistical Significance (ANOVA):")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   P-value: {p_anova:.4e}")
    print(f"   Significant difference: {'❌ Yes' if p_anova < 0.05 else '✅ No'}")
    
    # Individual t-tests
    t_mf, p_mf = stats.ttest_ind(male_salaries, female_salaries)
    t_mo, p_mo = stats.ttest_ind(male_salaries, other_salaries)
    t_fo, p_fo = stats.ttest_ind(female_salaries, other_salaries)
    
    print(f"\n🔬 Pairwise T-tests:")
    print(f"   Male vs Female: p={p_mf:.4e} {'❌ Significant' if p_mf < 0.05 else '✅ Not significant'}")
    print(f"   Male vs Other: p={p_mo:.4e} {'❌ Significant' if p_mo < 0.05 else '✅ Not significant'}")
    print(f"   Female vs Other: p={p_fo:.4e} {'❌ Significant' if p_fo < 0.05 else '✅ Not significant'}")
    
    return {
        'gender_dist': gender_dist,
        'pay_gaps': {
            'male_female': male_female_gap,
            'male_other': male_other_gap,
            'female_other': female_other_gap
        },
        'p_values': {
            'anova': p_anova,
            'male_female': p_mf,
            'male_other': p_mo,
            'female_other': p_fo
        }
    }

def analyze_role_equity(df):
    """Analyze equity across job roles"""
    print("\n" + "="*60)
    print("👔 JOB ROLE EQUITY ANALYSIS")
    print("="*60)
    
    # Gender distribution by role
    role_gender = pd.crosstab(df['job_role'], df['gender'], normalize='index') * 100
    print(f"\n📊 Gender Distribution by Job Role (%):")
    print(role_gender.round(2))
    
    # Check for roles with severe gender imbalance
    print(f"\n⚠️ Roles with Gender Imbalance (>60% single gender):")
    imbalanced_roles = []
    for role in role_gender.index:
        max_pct = role_gender.loc[role].max()
        if max_pct > 60:
            dominant_gender = role_gender.loc[role].idxmax()
            imbalanced_roles.append({
                'role': role,
                'dominant_gender': dominant_gender,
                'percentage': max_pct
            })
            print(f"   {role}: {max_pct:.1f}% {dominant_gender}")
    
    if not imbalanced_roles:
        print("   ✅ No severely imbalanced roles found!")
    
    # Salary equity by role and gender
    print(f"\n💰 Average Salary by Role and Gender:")
    role_salary_gender = df.groupby(['job_role', 'gender'])['salary'].mean().unstack()
    print(role_salary_gender.round(0))
    
    # Calculate pay gaps within each role
    role_gaps = []
    for role in df['job_role'].unique():
        role_data = df[df['job_role'] == role]
        male_data = role_data[role_data['gender'] == 'Male']['salary']
        female_data = role_data[role_data['gender'] == 'Female']['salary']
        
        if len(male_data) > 10 and len(female_data) > 10:
            gap = ((male_data.mean() - female_data.mean()) / female_data.mean()) * 100
            # T-test for significance
            t_stat, p_val = stats.ttest_ind(male_data, female_data)
            
            role_gaps.append({
                'role': role,
                'gap_percent': gap,
                'p_value': p_val,
                'male_count': len(male_data),
                'female_count': len(female_data),
                'significant': p_val < 0.05
            })
    
    role_gaps_df = pd.DataFrame(role_gaps)
    if not role_gaps_df.empty:
        print(f"\n📉 Pay Gaps by Job Role (Male vs Female):")
        for _, row in role_gaps_df.sort_values('gap_percent', key=abs, ascending=False).head(10).iterrows():
            sig_marker = "❌" if row['significant'] else "✅"
            print(f"   {sig_marker} {row['role']}: {row['gap_percent']:+.2f}% (p={row['p_value']:.3f})")
    
    return role_gaps_df

def analyze_experience_equity(df):
    """Analyze equity across experience levels"""
    print("\n" + "="*60)
    print("📅 EXPERIENCE LEVEL EQUITY ANALYSIS")
    print("="*60)
    
    # Create experience bins
    df['exp_bin'] = pd.cut(df['experience_years'], 
                          bins=[0, 3, 7, 12, 20, 50], 
                          labels=['0-3y', '4-7y', '8-12y', '13-20y', '20+y'])
    
    # Gender distribution by experience
    exp_gender = pd.crosstab(df['exp_bin'], df['gender'], normalize='index') * 100
    print(f"\n📊 Gender Distribution by Experience Level (%):")
    print(exp_gender.round(2))
    
    # Salary by experience and gender
    exp_salary = df.groupby(['exp_bin', 'gender'])['salary'].mean().unstack()
    print(f"\n💰 Average Salary by Experience and Gender:")
    print(exp_salary.round(0))
    
    # Check for experience-based pay gaps
    print(f"\n📈 Pay Gaps by Experience Level (Male vs Female):")
    for exp_level in df['exp_bin'].dropna().unique():
        exp_data = df[df['exp_bin'] == exp_level]
        male_salaries = exp_data[exp_data['gender'] == 'Male']['salary']
        female_salaries = exp_data[exp_data['gender'] == 'Female']['salary']
        
        if len(male_salaries) > 5 and len(female_salaries) > 5:
            gap = ((male_salaries.mean() - female_salaries.mean()) / female_salaries.mean()) * 100
            t_stat, p_val = stats.ttest_ind(male_salaries, female_salaries)
            sig_marker = "❌" if p_val < 0.05 else "✅"
            print(f"   {sig_marker} {exp_level}: {gap:+.2f}% (p={p_val:.3f})")

def calculate_bias_score(df):
    """Calculate overall bias score for the dataset"""
    print("\n" + "="*60)
    print("🎯 OVERALL BIAS SCORE CALCULATION")
    print("="*60)
    
    bias_components = {}
    
    # 1. Gender representation bias (0-25 points)
    gender_counts = df['gender'].value_counts()
    total = len(df)
    expected = total / len(gender_counts)
    
    representation_bias = sum([abs(count - expected) for count in gender_counts]) / expected
    rep_score = min(representation_bias * 5, 25)
    bias_components['representation'] = rep_score
    
    # 2. Overall salary bias (0-30 points)
    male_avg = df[df['gender'] == 'Male']['salary'].mean()
    female_avg = df[df['gender'] == 'Female']['salary'].mean()
    other_avg = df[df['gender'] == 'Other']['salary'].mean()
    
    salary_gap = max(
        abs((male_avg - female_avg) / female_avg * 100),
        abs((male_avg - other_avg) / other_avg * 100),
        abs((female_avg - other_avg) / other_avg * 100)
    )
    salary_score = min(salary_gap * 3, 30)
    bias_components['salary'] = salary_score
    
    # 3. Role distribution bias (0-25 points)
    role_gender_dist = pd.crosstab(df['job_role'], df['gender'], normalize='index')
    role_bias_scores = []
    
    for role in role_gender_dist.index:
        max_pct = role_gender_dist.loc[role].max()
        min_pct = role_gender_dist.loc[role].min()
        role_bias = (max_pct - min_pct) * 100
        role_bias_scores.append(role_bias)
    
    avg_role_bias = np.mean(role_bias_scores)
    role_score = min(avg_role_bias * 0.5, 25)
    bias_components['role_distribution'] = role_score
    
    # 4. Statistical significance penalty (0-20 points)
    # Test if salary differences are statistically significant
    male_salaries = df[df['gender'] == 'Male']['salary']
    female_salaries = df[df['gender'] == 'Female']['salary']
    other_salaries = df[df['gender'] == 'Other']['salary']
    
    f_stat, p_anova = stats.f_oneway(male_salaries, female_salaries, other_salaries)
    significance_score = 20 if p_anova < 0.001 else (10 if p_anova < 0.05 else 0)
    bias_components['significance'] = significance_score
    
    # Calculate total bias score
    total_bias = sum(bias_components.values())
    
    print(f"📊 Bias Score Breakdown:")
    print(f"   Representation Bias: {rep_score:.1f}/25")
    print(f"   Salary Gap Bias: {salary_score:.1f}/30") 
    print(f"   Role Distribution Bias: {role_score:.1f}/25")
    print(f"   Statistical Significance: {significance_score:.1f}/20")
    print(f"   TOTAL BIAS SCORE: {total_bias:.1f}/100")
    
    # Bias level assessment
    if total_bias < 10:
        level = "🟢 VERY LOW BIAS"
        assessment = "Excellent equity"
    elif total_bias < 25:
        level = "🟡 LOW BIAS" 
        assessment = "Good equity with minor issues"
    elif total_bias < 50:
        level = "🟠 MODERATE BIAS"
        assessment = "Some equity concerns"
    else:
        level = "🔴 HIGH BIAS"
        assessment = "Significant equity issues"
    
    print(f"\n🎯 BIAS ASSESSMENT: {level}")
    print(f"📋 Summary: {assessment}")
    
    return total_bias, bias_components

def create_equity_visualizations(df):
    """Create visualizations for equity analysis"""
    print("\n📊 Creating equity visualizations...")
    
    # Set style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Salary Dataset Equity Analysis (Without Bias)', fontsize=16, fontweight='bold')
    
    # 1. Gender distribution pie chart
    gender_counts = df['gender'].value_counts()
    axes[0, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.2f%%', startangle=90)
    axes[0, 0].set_title('Gender Distribution')
    
    # 2. Salary distribution by gender
    for gender in df['gender'].unique():
        gender_salaries = df[df['gender'] == gender]['salary']
        axes[0, 1].hist(gender_salaries, alpha=0.6, label=gender, bins=30)
    axes[0, 1].set_title('Salary Distribution by Gender')
    axes[0, 1].set_xlabel('Salary')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Average salary by gender with error bars
    gender_stats = df.groupby('gender')['salary'].agg(['mean', 'std'])
    bars = axes[0, 2].bar(gender_stats.index, gender_stats['mean'], 
                         yerr=gender_stats['std'], capsize=5, alpha=0.7)
    axes[0, 2].set_title('Average Salary by Gender (with std dev)')
    axes[0, 2].set_ylabel('Average Salary')
    
    # Add value labels
    for i, (bar, mean_val) in enumerate(zip(bars, gender_stats['mean'])):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + gender_stats['std'].iloc[i],
                       f'${mean_val:,.0f}', ha='center', va='bottom')
    
    # 4. Gender distribution by job role
    role_gender = pd.crosstab(df['job_role'], df['gender'], normalize='index')
    role_gender.plot(kind='bar', stacked=True, ax=axes[1, 0], width=0.8)
    axes[1, 0].set_title('Gender Distribution by Job Role')
    axes[1, 0].set_xlabel('Job Role')
    axes[1, 0].set_ylabel('Proportion')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend(title='Gender')
    
    # 5. Salary by experience and gender
    df['exp_bin'] = pd.cut(df['experience_years'], bins=[0, 3, 7, 12, 20, 50], 
                          labels=['0-3y', '4-7y', '8-12y', '13-20y', '20+y'])
    
    exp_salary = df.groupby(['exp_bin', 'gender'])['salary'].mean().unstack()
    
    x_pos = np.arange(len(exp_salary.index))
    width = 0.25
    
    for i, gender in enumerate(exp_salary.columns):
        if not exp_salary[gender].isna().all():
            axes[1, 1].bar(x_pos + i*width, exp_salary[gender], width, 
                          label=gender, alpha=0.8)
    
    axes[1, 1].set_title('Average Salary by Experience and Gender')
    axes[1, 1].set_xlabel('Experience Level')
    axes[1, 1].set_ylabel('Average Salary')
    axes[1, 1].set_xticks(x_pos + width)
    axes[1, 1].set_xticklabels(exp_salary.index)
    axes[1, 1].legend()
    
    # 6. Salary range comparison
    salary_ranges = df.groupby('gender')['salary'].agg(['min', 'max', 'mean'])
    
    x_pos = np.arange(len(salary_ranges.index))
    axes[1, 2].bar(x_pos, salary_ranges['max'] - salary_ranges['min'], 
                   bottom=salary_ranges['min'], alpha=0.6, label='Salary Range')
    axes[1, 2].scatter(x_pos, salary_ranges['mean'], color='red', s=100, 
                      label='Average', zorder=5)
    
    axes[1, 2].set_title('Salary Range by Gender')
    axes[1, 2].set_xlabel('Gender')
    axes[1, 2].set_ylabel('Salary')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(salary_ranges.index)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('without_bias_equity_analysis.png', dpi=300, bbox_inches='tight')
    print("📈 Visualizations saved as 'without_bias_equity_analysis.png'")
    plt.show()

def generate_equity_report(df, bias_score, bias_components):
    """Generate final equity assessment report"""
    print("\n" + "="*60)
    print("📋 FINAL EQUITY ASSESSMENT REPORT")
    print("="*60)
    
    total_records = len(df)
    
    # Gender statistics
    gender_counts = df['gender'].value_counts()
    male_pct = (gender_counts.get('Male', 0) / total_records) * 100
    female_pct = (gender_counts.get('Female', 0) / total_records) * 100
    other_pct = (gender_counts.get('Other', 0) / total_records) * 100
    
    # Salary statistics
    male_avg = df[df['gender'] == 'Male']['salary'].mean()
    female_avg = df[df['gender'] == 'Female']['salary'].mean() 
    other_avg = df[df['gender'] == 'Other']['salary'].mean()
    
    overall_gap = abs((male_avg - female_avg) / female_avg * 100)
    
    print(f"📊 DATASET SUMMARY:")
    print(f"   Total Records: {total_records:,}")
    print(f"   Gender Split: Male {male_pct:.2f}%, Female {female_pct:.2f}%, Other {other_pct:.2f}%")
    print(f"   Salary Range: ${df['salary'].min():,} - ${df['salary'].max():,}")
    
    print(f"\n💰 EQUITY METRICS:")
    print(f"   Male Average: ${male_avg:,.2f}")
    print(f"   Female Average: ${female_avg:,.2f}")
    print(f"   Other Average: ${other_avg:,.2f}")
    print(f"   Max Pay Gap: {overall_gap:.3f}%")
    
    print(f"\n🎯 BIAS ASSESSMENT:")
    print(f"   Overall Bias Score: {bias_score:.1f}/100")
    
    if bias_score < 10:
        print(f"   🟢 VERDICT: EXCELLENT EQUITY")
        print(f"   This dataset demonstrates exceptional gender equity")
    elif bias_score < 25:
        print(f"   🟡 VERDICT: GOOD EQUITY")
        print(f"   This dataset shows good gender equity with minor areas for improvement")
    elif bias_score < 50:
        print(f"   🟠 VERDICT: MODERATE EQUITY")
        print(f"   This dataset has some equity concerns that should be addressed")
    else:
        print(f"   🔴 VERDICT: POOR EQUITY")
        print(f"   This dataset has significant equity issues requiring immediate attention")
    
    print(f"\n📈 KEY FINDINGS:")
    if bias_components['representation'] < 5:
        print(f"   ✅ Excellent gender representation balance")
    else:
        print(f"   ⚠️ Gender representation could be more balanced")
    
    if bias_components['salary'] < 5:
        print(f"   ✅ Minimal gender-based salary gaps")
    else:
        print(f"   ⚠️ Notable salary gaps exist between genders")
        
    if bias_components['role_distribution'] < 10:
        print(f"   ✅ Good gender distribution across job roles")
    else:
        print(f"   ⚠️ Some job roles show gender concentration")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if bias_score < 15:
        print(f"   • Continue current equitable practices")
        print(f"   • Maintain regular monitoring")
    else:
        print(f"   • Review salary setting processes")
        print(f"   • Implement structured hiring practices")
        print(f"   • Provide bias training for decision makers")
    
    print(f"   • Regular equity audits and reporting")
    print(f"   • Transparent compensation frameworks")

def main():
    """Main function for comprehensive bias analysis"""
    print("🔍 COMPREHENSIVE BIAS ANALYSIS")
    print("Dataset: salary_prediction_dataset_100k_without_bias.csv")
    print("=" * 70)
    
    # Load data
    df = load_unbiased_dataset()
    if df is None:
        return
    
    # Run comprehensive analysis
    gender_results = analyze_gender_equity(df)
    role_results = analyze_role_equity(df)
    analyze_experience_equity(df)
    
    # Calculate bias score
    bias_score, bias_components = calculate_bias_score(df)
    
    # Create visualizations
    try:
        create_equity_visualizations(df)
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")
    
    # Generate final report
    generate_equity_report(df, bias_score, bias_components)
    
    print(f"\n🎉 Comprehensive bias analysis completed!")
    print(f"💡 This dataset shows a bias score of {bias_score:.1f}/100")

if __name__ == "__main__":
    main()
