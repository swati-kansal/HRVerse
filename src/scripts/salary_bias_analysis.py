#!/usr/bin/env python3
"""
Salary Dataset Bias Analysis
Comprehensive bias analysis for salary prediction dataset with gender information
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_salary_data():
    """Load the salary dataset"""
    try:
        print("📥 Loading salary prediction dataset...")
        df = pd.read_csv('src/storage/salary_prediction_dataset_100k_with_gender.csv')
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def basic_statistics(df):
    """Calculate basic statistics about the dataset"""
    print("\n" + "="*60)
    print("📈 BASIC DATASET STATISTICS")
    print("="*60)
    
    print(f"Total records: {len(df):,}")
    print(f"Features: {len(df.columns)}")
    
    # Gender distribution
    print(f"\n👥 Gender Distribution:")
    gender_counts = df['gender'].value_counts()
    for gender, count in gender_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {gender}: {count:,} ({percentage:.2f}%)")
    
    # Salary statistics
    print(f"\n💰 Salary Statistics:")
    print(f"   Mean salary: ${df['salary'].mean():,.2f}")
    print(f"   Median salary: ${df['salary'].median():,.2f}")
    print(f"   Min salary: ${df['salary'].min():,.2f}")
    print(f"   Max salary: ${df['salary'].max():,.2f}")
    print(f"   Std deviation: ${df['salary'].std():,.2f}")

def gender_salary_bias_analysis(df):
    """Analyze salary bias by gender"""
    print("\n" + "="*60)
    print("⚖️ GENDER-BASED SALARY BIAS ANALYSIS")
    print("="*60)
    
    # Salary by gender
    gender_salary_stats = df.groupby('gender')['salary'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    print("\n📊 Salary Statistics by Gender:")
    print(gender_salary_stats)
    
    # Calculate salary gaps
    male_avg = df[df['gender'] == 'Male']['salary'].mean()
    female_avg = df[df['gender'] == 'Female']['salary'].mean()
    other_avg = df[df['gender'] == 'Other']['salary'].mean()
    
    print(f"\n💰 Average Salaries:")
    print(f"   Male: ${male_avg:,.2f}")
    print(f"   Female: ${female_avg:,.2f}")
    print(f"   Other: ${other_avg:,.2f}")
    
    # Gender pay gaps
    male_female_gap = ((male_avg - female_avg) / female_avg) * 100
    male_other_gap = ((male_avg - other_avg) / other_avg) * 100
    female_other_gap = ((female_avg - other_avg) / other_avg) * 100
    
    print(f"\n📉 Pay Gaps (as percentage):")
    print(f"   Male vs Female: {male_female_gap:+.2f}%")
    print(f"   Male vs Other: {male_other_gap:+.2f}%")
    print(f"   Female vs Other: {female_other_gap:+.2f}%")
    
    # Statistical significance test
    male_salaries = df[df['gender'] == 'Male']['salary']
    female_salaries = df[df['gender'] == 'Female']['salary']
    other_salaries = df[df['gender'] == 'Other']['salary']
    
    # T-test between male and female
    t_stat_mf, p_value_mf = stats.ttest_ind(male_salaries, female_salaries)
    
    print(f"\n🔬 Statistical Significance Tests:")
    print(f"   Male vs Female t-test:")
    print(f"     t-statistic: {t_stat_mf:.4f}")
    print(f"     p-value: {p_value_mf:.4e}")
    print(f"     Significant: {'Yes' if p_value_mf < 0.05 else 'No'}")
    
    return gender_salary_stats

def positional_bias_analysis(df):
    """Analyze bias by job roles and positions"""
    print("\n" + "="*60)
    print("🏢 POSITIONAL BIAS ANALYSIS")
    print("="*60)
    
    # Gender distribution by job role
    job_gender_dist = pd.crosstab(df['job_role'], df['gender'], normalize='index') * 100
    print("\n👔 Gender Distribution by Job Role (%):")
    print(job_gender_dist.round(2))
    
    # Average salary by job role and gender
    job_salary_gender = df.groupby(['job_role', 'gender'])['salary'].mean().unstack()
    print(f"\n💼 Average Salary by Job Role and Gender:")
    print(job_salary_gender.round(0))
    
    # Identify roles with highest gender disparities
    job_roles = df['job_role'].unique()
    disparities = []
    
    for role in job_roles:
        role_data = df[df['job_role'] == role]
        if len(role_data['gender'].unique()) >= 2:
            male_data = role_data[role_data['gender'] == 'Male']['salary']
            female_data = role_data[role_data['gender'] == 'Female']['salary']
            
            if len(male_data) > 0 and len(female_data) > 0:
                gap = ((male_data.mean() - female_data.mean()) / female_data.mean()) * 100
                disparities.append({
                    'job_role': role,
                    'male_avg': male_data.mean(),
                    'female_avg': female_data.mean(),
                    'pay_gap_percent': gap,
                    'male_count': len(male_data),
                    'female_count': len(female_data)
                })
    
    disparity_df = pd.DataFrame(disparities).sort_values('pay_gap_percent', ascending=False)
    
    print(f"\n⚠️ Job Roles with Highest Gender Pay Gaps:")
    print(disparity_df[['job_role', 'pay_gap_percent', 'male_count', 'female_count']].head(10))
    
    return disparity_df

def experience_bias_analysis(df):
    """Analyze bias related to experience levels"""
    print("\n" + "="*60)
    print("📅 EXPERIENCE-BASED BIAS ANALYSIS")
    print("="*60)
    
    # Create experience bins
    df['experience_bin'] = pd.cut(df['experience_years'], 
                                 bins=[0, 3, 7, 12, 20, 50], 
                                 labels=['0-3 years', '4-7 years', '8-12 years', '13-20 years', '20+ years'])
    
    # Gender distribution by experience level
    exp_gender_dist = pd.crosstab(df['experience_bin'], df['gender'], normalize='index') * 100
    print("\n📊 Gender Distribution by Experience Level (%):")
    print(exp_gender_dist.round(2))
    
    # Salary by experience and gender
    exp_salary_gender = df.groupby(['experience_bin', 'gender'])['salary'].mean().unstack()
    print(f"\n💰 Average Salary by Experience and Gender:")
    print(exp_salary_gender.round(0))
    
    # Calculate pay gaps by experience level
    print(f"\n📈 Pay Gaps by Experience Level:")
    for exp_level in df['experience_bin'].unique():
        if pd.isna(exp_level):
            continue
        exp_data = df[df['experience_bin'] == exp_level]
        male_salaries = exp_data[exp_data['gender'] == 'Male']['salary']
        female_salaries = exp_data[exp_data['gender'] == 'Female']['salary']
        
        if len(male_salaries) > 0 and len(female_salaries) > 0:
            gap = ((male_salaries.mean() - female_salaries.mean()) / female_salaries.mean()) * 100
            print(f"   {exp_level}: {gap:+.2f}%")

def education_bias_analysis(df):
    """Analyze bias related to education levels"""
    print("\n" + "="*60)
    print("🎓 EDUCATION-BASED BIAS ANALYSIS")
    print("="*60)
    
    # Gender distribution by education level
    edu_gender_dist = pd.crosstab(df['education_level'], df['gender'], normalize='index') * 100
    print("\n📚 Gender Distribution by Education Level (%):")
    print(edu_gender_dist.round(2))
    
    # Salary by education and gender
    edu_salary_gender = df.groupby(['education_level', 'gender'])['salary'].mean().unstack()
    print(f"\n🎯 Average Salary by Education and Gender:")
    print(edu_salary_gender.round(0))

def industry_bias_analysis(df):
    """Analyze bias across different industries"""
    print("\n" + "="*60)
    print("🏭 INDUSTRY-BASED BIAS ANALYSIS")
    print("="*60)
    
    # Gender distribution by industry
    industry_gender_dist = pd.crosstab(df['industry'], df['gender'], normalize='index') * 100
    print("\n🏢 Gender Distribution by Industry (%):")
    print(industry_gender_dist.round(2))
    
    # Salary by industry and gender
    industry_salary_gender = df.groupby(['industry', 'gender'])['salary'].mean().unstack()
    print(f"\n💼 Average Salary by Industry and Gender:")
    print(industry_salary_gender.round(0))
    
    # Industry pay gaps
    industries = df['industry'].unique()
    industry_gaps = []
    
    for industry in industries:
        industry_data = df[df['industry'] == industry]
        male_data = industry_data[industry_data['gender'] == 'Male']['salary']
        female_data = industry_data[industry_data['gender'] == 'Female']['salary']
        
        if len(male_data) > 0 and len(female_data) > 0:
            gap = ((male_data.mean() - female_data.mean()) / female_data.mean()) * 100
            industry_gaps.append({
                'industry': industry,
                'pay_gap_percent': gap,
                'male_count': len(male_data),
                'female_count': len(female_data)
            })
    
    industry_gap_df = pd.DataFrame(industry_gaps).sort_values('pay_gap_percent', ascending=False)
    print(f"\n⚠️ Industries with Highest Gender Pay Gaps:")
    print(industry_gap_df[['industry', 'pay_gap_percent']].head())

def create_bias_visualizations(df):
    """Create visualizations for bias analysis"""
    print("\n" + "="*60)
    print("📊 CREATING BIAS VISUALIZATIONS")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Salary Dataset Bias Analysis', fontsize=16, fontweight='bold')
    
    # 1. Salary distribution by gender
    axes[0, 0].hist([df[df['gender'] == 'Male']['salary'],
                     df[df['gender'] == 'Female']['salary'],
                     df[df['gender'] == 'Other']['salary']], 
                   label=['Male', 'Female', 'Other'], alpha=0.7, bins=30)
    axes[0, 0].set_title('Salary Distribution by Gender')
    axes[0, 0].set_xlabel('Salary')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # 2. Average salary by gender
    gender_avg_salary = df.groupby('gender')['salary'].mean()
    bars = axes[0, 1].bar(gender_avg_salary.index, gender_avg_salary.values)
    axes[0, 1].set_title('Average Salary by Gender')
    axes[0, 1].set_ylabel('Average Salary')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:,.0f}', ha='center', va='bottom')
    
    # 3. Gender distribution
    gender_counts = df['gender'].value_counts()
    axes[0, 2].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    axes[0, 2].set_title('Gender Distribution in Dataset')
    
    # 4. Salary by experience and gender
    exp_salary = df.groupby(['experience_years', 'gender'])['salary'].mean().unstack()
    if 'Male' in exp_salary.columns and 'Female' in exp_salary.columns:
        axes[1, 0].plot(exp_salary.index, exp_salary['Male'], 'o-', label='Male')
        axes[1, 0].plot(exp_salary.index, exp_salary['Female'], 's-', label='Female')
        if 'Other' in exp_salary.columns:
            axes[1, 0].plot(exp_salary.index, exp_salary['Other'], '^-', label='Other')
    axes[1, 0].set_title('Salary vs Experience by Gender')
    axes[1, 0].set_xlabel('Experience (Years)')
    axes[1, 0].set_ylabel('Average Salary')
    axes[1, 0].legend()
    
    # 5. Salary by job role (top 10)
    top_roles = df['job_role'].value_counts().head(10).index
    role_salary = df[df['job_role'].isin(top_roles)].groupby('job_role')['salary'].mean()
    axes[1, 1].barh(role_salary.index, role_salary.values)
    axes[1, 1].set_title('Average Salary by Job Role (Top 10)')
    axes[1, 1].set_xlabel('Average Salary')
    
    # 6. Gender distribution by industry
    industry_gender = pd.crosstab(df['industry'], df['gender'], normalize='index')
    industry_gender.plot(kind='bar', stacked=True, ax=axes[1, 2])
    axes[1, 2].set_title('Gender Distribution by Industry')
    axes[1, 2].set_xlabel('Industry')
    axes[1, 2].set_ylabel('Proportion')
    axes[1, 2].legend(title='Gender')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('salary_bias_analysis.png', dpi=300, bbox_inches='tight')
    print("📈 Visualizations saved as 'salary_bias_analysis.png'")
    plt.show()

def bias_summary_report(df, gender_stats, job_disparities):
    """Generate a comprehensive bias summary report"""
    print("\n" + "="*60)
    print("📋 BIAS SUMMARY REPORT")
    print("="*60)
    
    # Overall dataset metrics
    total_records = len(df)
    male_pct = (len(df[df['gender'] == 'Male']) / total_records) * 100
    female_pct = (len(df[df['gender'] == 'Female']) / total_records) * 100
    other_pct = (len(df[df['gender'] == 'Other']) / total_records) * 100
    
    # Salary statistics
    male_avg = df[df['gender'] == 'Male']['salary'].mean()
    female_avg = df[df['gender'] == 'Female']['salary'].mean()
    overall_gap = ((male_avg - female_avg) / female_avg) * 100
    
    print(f"📊 DATASET OVERVIEW:")
    print(f"   Total Records: {total_records:,}")
    print(f"   Gender Distribution: Male {male_pct:.1f}%, Female {female_pct:.1f}%, Other {other_pct:.1f}%")
    
    print(f"\n💰 SALARY BIAS FINDINGS:")
    print(f"   Overall Male-Female Pay Gap: {overall_gap:+.2f}%")
    print(f"   Male Average Salary: ${male_avg:,.2f}")
    print(f"   Female Average Salary: ${female_avg:,.2f}")
    
    print(f"\n⚠️ BIAS INDICATORS:")
    
    # Check for significant bias
    if abs(overall_gap) > 5:
        print(f"   🔴 SIGNIFICANT GENDER PAY GAP: {overall_gap:+.2f}%")
    else:
        print(f"   🟢 Gender pay gap within acceptable range: {overall_gap:+.2f}%")
    
    # Check gender representation
    if min(male_pct, female_pct) < 30:
        print(f"   🔴 GENDER UNDERREPRESENTATION: Minority gender <30%")
    else:
        print(f"   🟢 Balanced gender representation")
    
    # Job role disparities
    if len(job_disparities) > 0:
        max_gap = job_disparities['pay_gap_percent'].max()
        worst_role = job_disparities.loc[job_disparities['pay_gap_percent'].idxmax(), 'job_role']
        print(f"   🔴 HIGHEST JOB ROLE GAP: {worst_role} ({max_gap:+.2f}%)")
    
    print(f"\n📈 RECOMMENDATIONS:")
    if abs(overall_gap) > 10:
        print(f"   • Conduct immediate salary equity review")
        print(f"   • Implement transparent pay scales")
        print(f"   • Review promotion and hiring practices")
    
    if min(male_pct, female_pct) < 20:
        print(f"   • Improve diversity in hiring")
        print(f"   • Review job posting and recruitment channels")
    
    print(f"   • Regular bias audits and monitoring")
    print(f"   • Bias training for managers and HR")
    print(f"   • Anonymous reporting systems for bias incidents")

def main():
    """Main function to run comprehensive bias analysis"""
    print("🔍 SALARY DATASET BIAS ANALYSIS")
    print("Comprehensive analysis of bias in salary prediction dataset")
    print("=" * 70)
    
    # Load data
    df = load_salary_data()
    if df is None:
        return
    
    # Run all analyses
    basic_statistics(df)
    gender_stats = gender_salary_bias_analysis(df)
    job_disparities = positional_bias_analysis(df)
    experience_bias_analysis(df)
    education_bias_analysis(df)
    industry_bias_analysis(df)
    
    # Create visualizations
    try:
        create_bias_visualizations(df)
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")
    
    # Generate summary report
    bias_summary_report(df, gender_stats, job_disparities)
    
    print(f"\n🎉 Bias analysis completed successfully!")
    print(f"💡 Review the findings above to identify and address potential biases")

if __name__ == "__main__":
    main()
