#!/usr/bin/env python3
"""
Simple Salary Bias Calculator
Easy-to-use script to calculate bias in any salary dataset with gender information
"""

import pandas as pd
import numpy as np

def calculate_salary_bias(csv_file_path):
    """
    Calculate bias in a salary dataset
    
    Args:
        csv_file_path: Path to CSV file with columns including 'gender' and 'salary'
    
    Returns:
        Dictionary with bias analysis results
    """
    
    try:
        # Load the dataset
        df = pd.read_csv(csv_file_path)
        print(f"📊 Analyzing: {csv_file_path}")
        print(f"📈 Records loaded: {len(df):,}")
        
        # Check required columns
        required_cols = ['gender', 'salary']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Basic statistics
        total_records = len(df)
        print(f"\n📋 BASIC ANALYSIS")
        print(f"{'='*40}")
        
        # Gender distribution
        gender_counts = df['gender'].value_counts()
        print(f"\n👥 Gender Distribution:")
        gender_percentages = {}
        for gender, count in gender_counts.items():
            pct = (count / total_records) * 100
            gender_percentages[gender] = pct
            print(f"   {gender}: {count:,} ({pct:.2f}%)")
        
        # Salary analysis by gender
        print(f"\n💰 Salary Analysis:")
        salary_by_gender = {}
        for gender in gender_counts.index:
            gender_salaries = df[df['gender'] == gender]['salary']
            avg_salary = gender_salaries.mean()
            salary_by_gender[gender] = avg_salary
            print(f"   {gender} average: ${avg_salary:,.0f}")
        
        # Calculate pay gaps
        print(f"\n📉 Pay Gap Analysis:")
        pay_gaps = {}
        
        genders = list(salary_by_gender.keys())
        max_gap = 0
        
        for i, gender1 in enumerate(genders):
            for gender2 in genders[i+1:]:
                salary1 = salary_by_gender[gender1]
                salary2 = salary_by_gender[gender2]
                
                # Calculate gap as percentage
                if salary2 != 0:
                    gap = ((salary1 - salary2) / salary2) * 100
                    pay_gaps[f"{gender1}_vs_{gender2}"] = gap
                    max_gap = max(max_gap, abs(gap))
                    print(f"   {gender1} vs {gender2}: {gap:+.2f}%")
        
        # Bias scoring
        print(f"\n🎯 BIAS SCORING")
        print(f"{'='*40}")
        
        # 1. Pay gap component (0-50 points)
        gap_score = min(max_gap * 1.5, 50)
        
        # 2. Representation component (0-30 points) 
        percentages = list(gender_percentages.values())
        if len(percentages) > 1:
            rep_imbalance = max(percentages) - min(percentages)
            rep_score = min(rep_imbalance * 2, 30)
        else:
            rep_score = 30  # Maximum penalty for no diversity
        
        # 3. Sample size component (0-20 points)
        min_gender_count = min(gender_counts.values)
        if min_gender_count < 100:
            sample_score = 20
        elif min_gender_count < 500:
            sample_score = 10
        else:
            sample_score = 0
        
        total_bias_score = gap_score + rep_score + sample_score
        
        print(f"📊 Score Breakdown:")
        print(f"   Pay Gap Score: {gap_score:.1f}/50 (Max gap: {max_gap:.2f}%)")
        print(f"   Representation Score: {rep_score:.1f}/30 (Imbalance: {rep_imbalance if len(percentages) > 1 else 'N/A'})")
        print(f"   Sample Size Score: {sample_score:.1f}/20 (Min group: {min_gender_count:,})")
        print(f"   TOTAL BIAS SCORE: {total_bias_score:.1f}/100")
        
        # Bias level assessment
        if total_bias_score < 10:
            bias_level = "🟢 VERY LOW BIAS"
            recommendation = "Excellent! Continue current practices."
        elif total_bias_score < 25:
            bias_level = "🟡 LOW BIAS"
            recommendation = "Good equity with room for minor improvements."
        elif total_bias_score < 50:
            bias_level = "🟠 MODERATE BIAS"
            recommendation = "Some equity concerns. Review compensation practices."
        elif total_bias_score < 75:
            bias_level = "🔴 HIGH BIAS"
            recommendation = "Significant equity issues. Immediate action needed."
        else:
            bias_level = "🚨 VERY HIGH BIAS"
            recommendation = "Critical equity problems. Comprehensive review required."
        
        print(f"\n🏆 OVERALL ASSESSMENT: {bias_level}")
        print(f"💡 Recommendation: {recommendation}")
        
        # Detailed insights
        print(f"\n📈 INSIGHTS & RECOMMENDATIONS")
        print(f"{'='*40}")
        
        if max_gap > 10:
            print(f"⚠️ Large pay gap detected ({max_gap:.1f}%)")
            print(f"   • Review salary setting processes")
            print(f"   • Implement structured compensation bands")
        elif max_gap < 2:
            print(f"✅ Excellent pay equity (gap only {max_gap:.1f}%)")
        
        if rep_imbalance > 20:
            print(f"⚠️ Gender representation imbalance detected")
            print(f"   • Review hiring and recruitment practices")
            print(f"   • Expand diversity initiatives")
        elif rep_imbalance < 5:
            print(f"✅ Good gender representation balance")
        
        if min_gender_count < 100:
            print(f"⚠️ Small sample size for some groups")
            print(f"   • Results may not be statistically robust")
            print(f"   • Collect more data for reliable analysis")
        
        print(f"\n🔄 General Recommendations:")
        print(f"   • Conduct regular bias audits")
        print(f"   • Implement transparent pay scales") 
        print(f"   • Provide bias training for managers")
        print(f"   • Use structured hiring processes")
        
        return {
            'file_path': csv_file_path,
            'total_records': total_records,
            'gender_distribution': gender_percentages,
            'salary_by_gender': salary_by_gender,
            'pay_gaps': pay_gaps,
            'max_pay_gap': max_gap,
            'bias_score': total_bias_score,
            'bias_level': bias_level,
            'recommendation': recommendation
        }
        
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file_path}")
        return None
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return None

def main():
    """Main function to analyze the without_bias dataset"""
    
    print("🔍 SALARY BIAS CALCULATOR")
    print("=" * 50)
    print("Analyzing: salary_prediction_dataset_100k_without_bias.csv")
    print("=" * 50)
    
    # Analyze the without_bias dataset
    result = calculate_salary_bias('src/storage/salary_prediction_dataset_100k_without_bias.csv')
    
    if result:
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📊 Dataset shows {result['bias_level']}")
        print(f"🎯 Final bias score: {result['bias_score']:.1f}/100")
    else:
        print(f"\n❌ Analysis failed!")

if __name__ == "__main__":
    main()
