import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from collections import Counter

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("storage/salary_predictor/salary_prediction_dataset_biased.csv")

# -----------------------------
# 1️⃣ BIAS DETECTION
# -----------------------------

print("="*60)
print("1️⃣ BIAS DETECTION IN ORIGINAL DATA")
print("="*60)

print("\n📊 GENDER DISTRIBUTION:")
print(df['gender'].value_counts())
print(f"Percentages: {df['gender'].value_counts(normalize=True).round(3).to_dict()}")

print("\n📊 COLLEGE DISTRIBUTION:")
print(df['college'].value_counts())
print(f"Percentages: {df['college'].value_counts(normalize=True).round(3).to_dict()}")

# -----------------------------
# Gender Salary Gap
# -----------------------------
gender_salary = df.groupby("gender")["salary"].mean()
print("\n💰 AVERAGE SALARY BY GENDER:")
print(gender_salary.round(2))
gender_gap = gender_salary["Male"] - gender_salary["Female"]
print(f"\n🔴 Gender Gap (Male - Female): ${gender_gap:,.2f}")

# T-test Gender
male_salary = df[df.gender == "Male"].salary
female_salary = df[df.gender == "Female"].salary
t_stat, p_val = ttest_ind(male_salary, female_salary, equal_var=False)
print(f"T-test p-value: {p_val:.2e} {'(Statistically Significant!)' if p_val < 0.05 else ''}")

# -----------------------------
# College Tier Bias
# -----------------------------
tier_salary = df.groupby("college")["salary"].mean()
print("\n💰 AVERAGE SALARY BY COLLEGE:")
print(tier_salary.round(2))
college_gap = tier_salary["Tier 1"] - tier_salary["Tier 3"]
print(f"\n🔴 College Gap (Tier 1 - Tier 3): ${college_gap:,.2f}")

# -----------------------------
# Visual Bias Detection
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x="gender", y="salary", data=df, ax=axes[0], order=["Male", "Female", "Other"])
axes[0].set_title("Salary Distribution by Gender (BIASED DATA)")
axes[0].set_ylabel("Salary ($)")

sns.boxplot(x="college", y="salary", data=df, ax=axes[1], order=["Tier 1", "Tier 2", "Tier 3"])
axes[1].set_title("Salary Distribution by College Tier (BIASED DATA)")
axes[1].set_ylabel("Salary ($)")

plt.tight_layout()
plt.savefig("salary_bias_analysis_biased_dataset.png", dpi=150)
plt.show()

# -----------------------------
# 2️⃣ BIAS MITIGATION PIPELINE
# -----------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("\n" + "="*60)
print("2️⃣ BIAS MITIGATION TECHNIQUES")
print("="*60)

# -----------------------------
# Helper Functions for Resampling by Sensitive Attributes
# -----------------------------

def undersample_to_balance(df, group_col):
    """Undersample majority groups to match the smallest group"""
    groups = df.groupby(group_col)
    min_size = groups.size().min()
    
    balanced_dfs = []
    for name, group in groups:
        sampled = group.sample(n=min_size, random_state=42)
        balanced_dfs.append(sampled)
    
    return pd.concat(balanced_dfs, ignore_index=True)

def oversample_to_balance(df, group_col):
    """Oversample minority groups to match the largest group"""
    groups = df.groupby(group_col)
    max_size = groups.size().max()
    
    balanced_dfs = []
    for name, group in groups:
        if len(group) < max_size:
            sampled = group.sample(n=max_size, replace=True, random_state=42)
        else:
            sampled = group.copy()
        balanced_dfs.append(sampled)
    
    return pd.concat(balanced_dfs, ignore_index=True)

def prepare_features(df_input):
    """Prepare features for model training - exclude sensitive attributes"""
    X = df_input.drop(columns=["salary", "gender", "college"], errors='ignore')
    X = pd.get_dummies(X)
    y = df_input["salary"]
    return X, y

# Store original feature columns for prediction alignment
X_original, y_original = prepare_features(df)
original_columns = X_original.columns

# -----------------------------
# BASELINE MODEL (No Mitigation)
# -----------------------------
print("\n" + "="*60)
print("BASELINE MODEL (No Bias Mitigation)")
print("="*60)

model_baseline = LinearRegression()
model_baseline.fit(X_original, y_original)
df["pred_baseline"] = model_baseline.predict(X_original)

print(f"Trained on: {len(df)} samples")
print(f"Gender distribution: {df['gender'].value_counts().to_dict()}")

# -----------------------------
# Technique 1: UNDERSAMPLING BY GENDER
# Reduce male samples to match female/other counts
# -----------------------------
print("\n" + "="*60)
print("TECHNIQUE 1: UNDERSAMPLING (Balance Gender Groups)")
print("="*60)

# Undersample by gender
df_undersampled = undersample_to_balance(df.copy(), "gender")
print(f"\nAfter Gender Undersampling:")
print(f"Total samples: {len(df_undersampled)} (was {len(df)})")
print(f"Gender distribution: {df_undersampled['gender'].value_counts().to_dict()}")
print(f"Salary by gender after undersampling:")
print(df_undersampled.groupby('gender')['salary'].mean())

# Prepare features and train
X_under, y_under = prepare_features(df_undersampled)
X_under = X_under.reindex(columns=original_columns, fill_value=0)

model_undersampled = LinearRegression()
model_undersampled.fit(X_under, y_under)

# Predict on original data
X_pred = X_original.reindex(columns=original_columns, fill_value=0)
df["pred_undersampled"] = model_undersampled.predict(X_pred)

# -----------------------------
# Technique 2: OVERSAMPLING BY GENDER
# Increase female/other samples to match male count
# -----------------------------
print("\n" + "="*60)
print("TECHNIQUE 2: OVERSAMPLING (Balance Gender Groups)")
print("="*60)

# Oversample by gender
df_oversampled = oversample_to_balance(df.copy(), "gender")
print(f"\nAfter Gender Oversampling:")
print(f"Total samples: {len(df_oversampled)} (was {len(df)})")
print(f"Gender distribution: {df_oversampled['gender'].value_counts().to_dict()}")
print(f"Salary by gender after oversampling:")
print(df_oversampled.groupby('gender')['salary'].mean())

# Prepare features and train
X_over, y_over = prepare_features(df_oversampled)
X_over = X_over.reindex(columns=original_columns, fill_value=0)

model_oversampled = LinearRegression()
model_oversampled.fit(X_over, y_over)

# Predict on original data
df["pred_oversampled"] = model_oversampled.predict(X_pred)

# -----------------------------
# Technique 3: SMOTE-like Synthetic Oversampling for Regression
# Generate synthetic samples for minority gender groups
# -----------------------------
print("\n" + "="*60)
print("TECHNIQUE 3: SMOTE-LIKE SYNTHETIC OVERSAMPLING")
print("="*60)

def smote_like_oversample(df, group_col, target_size=None):
    """
    SMOTE-like oversampling for regression:
    Creates synthetic samples by interpolating between existing samples
    """
    groups = df.groupby(group_col)
    if target_size is None:
        target_size = groups.size().max()
    
    balanced_dfs = []
    for name, group in groups:
        if len(group) >= target_size:
            balanced_dfs.append(group.copy())
        else:
            # Keep original samples
            balanced_dfs.append(group.copy())
            
            # Generate synthetic samples
            n_synthetic = target_size - len(group)
            synthetic_samples = []
            
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            categorical_cols = group.select_dtypes(exclude=[np.number]).columns
            
            for _ in range(n_synthetic):
                # Pick two random samples
                idx1, idx2 = np.random.choice(len(group), 2, replace=False)
                sample1 = group.iloc[idx1]
                sample2 = group.iloc[idx2]
                
                # Interpolation factor
                alpha = np.random.uniform(0.3, 0.7)
                
                # Create synthetic sample
                synthetic = {}
                for col in numeric_cols:
                    synthetic[col] = sample1[col] * alpha + sample2[col] * (1 - alpha)
                    if col in ['experience_years', 'education_level', 'skills_count', 'salary']:
                        synthetic[col] = int(round(synthetic[col]))
                
                for col in categorical_cols:
                    synthetic[col] = sample1[col] if np.random.random() < 0.5 else sample2[col]
                
                synthetic_samples.append(synthetic)
            
            synthetic_df = pd.DataFrame(synthetic_samples)
            balanced_dfs.append(synthetic_df)
    
    return pd.concat(balanced_dfs, ignore_index=True)

# Apply SMOTE-like oversampling
df_smote = smote_like_oversample(df.copy(), "gender")
print(f"\nAfter SMOTE-like Oversampling:")
print(f"Total samples: {len(df_smote)} (was {len(df)})")
print(f"Gender distribution: {df_smote['gender'].value_counts().to_dict()}")
print(f"Salary by gender after SMOTE:")
print(df_smote.groupby('gender')['salary'].mean())

# Prepare features and train
X_smote, y_smote = prepare_features(df_smote)
X_smote = X_smote.reindex(columns=original_columns, fill_value=0)

model_smote = LinearRegression()
model_smote.fit(X_smote, y_smote)

# Predict on original data
df["pred_smote"] = model_smote.predict(X_pred)

# -----------------------------
# Technique 4: COMBINED UNDERSAMPLING (Gender + College)
# Balance both sensitive attributes
# -----------------------------
print("\n" + "="*60)
print("TECHNIQUE 4: COMBINED UNDERSAMPLING (Gender + College)")
print("="*60)

# First undersample by gender, then by college
df_combined = undersample_to_balance(df.copy(), "gender")
df_combined = undersample_to_balance(df_combined, "college")
print(f"\nAfter Combined Undersampling:")
print(f"Total samples: {len(df_combined)} (was {len(df)})")
print(f"Gender distribution: {df_combined['gender'].value_counts().to_dict()}")
print(f"College distribution: {df_combined['college'].value_counts().to_dict()}")

# Prepare features and train
X_comb, y_comb = prepare_features(df_combined)
X_comb = X_comb.reindex(columns=original_columns, fill_value=0)

model_combined = LinearRegression()
model_combined.fit(X_comb, y_comb)

# Predict on original data
df["pred_combined_under"] = model_combined.predict(X_pred)

# -----------------------------
# Technique 5: Preprocessing – Reweighing
# -----------------------------
print("\n" + "="*60)
print("TECHNIQUE 5: REWEIGHING")
print("="*60)

def compute_weights(df, sensitive_attr):
    probs = df[sensitive_attr].value_counts(normalize=True)
    weights = df[sensitive_attr].map(lambda x: 1 / probs[x])
    weights = weights / weights.mean()
    return weights

df["gender_weight"] = compute_weights(df, "gender")
df["college_weight"] = compute_weights(df, "college")
df["final_weight"] = (df["gender_weight"] + df["college_weight"]) / 2

print(f"Weight statistics:")
print(f"Gender weights: {df.groupby('gender')['gender_weight'].mean().to_dict()}")
print(f"College weights: {df.groupby('college')['college_weight'].mean().to_dict()}")

# Train model with fairness weights
model_reweighed = LinearRegression()
model_reweighed.fit(X_original, y_original, sample_weight=df["final_weight"])
df["pred_reweighed"] = model_reweighed.predict(X_original)

# -----------------------------
# Technique 6: Post-processing – Salary Adjustment
# -----------------------------
print("\n" + "="*60)
print("TECHNIQUE 6: POST-PROCESSING ADJUSTMENT")
print("="*60)

# Calculate actual gaps to determine adjustment factors
male_avg = df[df['gender'] == 'Male']['salary'].mean()
female_avg = df[df['gender'] == 'Female']['salary'].mean()
gender_gap_pct = (male_avg - female_avg) / female_avg

tier1_avg = df[df['college'] == 'Tier 1']['salary'].mean()
tier3_avg = df[df['college'] == 'Tier 3']['salary'].mean()
college_gap_pct = (tier1_avg - tier3_avg) / tier3_avg

print(f"Detected Gender Gap: {gender_gap_pct*100:.1f}%")
print(f"Detected College Gap: {college_gap_pct*100:.1f}%")

def debias_salary(pred, gender, college):
    adj = 0
    # Adjust based on detected gaps
    if gender == "Female":
        adj += pred * (gender_gap_pct * 0.5)  # Partial correction
    elif gender == "Other":
        adj += pred * (gender_gap_pct * 0.3)
    
    if college == "Tier 3":
        adj += pred * (college_gap_pct * 0.3)
    elif college == "Tier 2":
        adj += pred * (college_gap_pct * 0.15)
    
    return pred + adj

df["pred_postprocessed"] = df.apply(
    lambda r: debias_salary(r["pred_baseline"], r["gender"], r["college"]),
    axis=1
)

# -----------------------------
# 3️⃣ COMPARISON OF ALL TECHNIQUES
# -----------------------------
print("\n" + "="*60)
print("3️⃣ COMPARISON OF ALL BIAS MITIGATION TECHNIQUES")
print("="*60)

# Calculate metrics for each technique
techniques = {
    "Baseline (No Mitigation)": "pred_baseline",
    "Undersampling (Gender)": "pred_undersampled",
    "Oversampling (Gender)": "pred_oversampled",
    "SMOTE-like (Gender)": "pred_smote",
    "Combined Undersampling": "pred_combined_under",
    "Reweighing": "pred_reweighed",
    "Post-processing": "pred_postprocessed"
}

results = []
for name, col in techniques.items():
    if col in df.columns and not df[col].isna().all():
        mse = mean_squared_error(y_original, df[col])
        r2 = r2_score(y_original, df[col])
        
        # Calculate gender bias (difference in mean predictions)
        male_pred = df[df["gender"] == "Male"][col].mean()
        female_pred = df[df["gender"] == "Female"][col].mean()
        gender_gap = male_pred - female_pred
        
        # Calculate college tier bias
        tier1_pred = df[df["college"] == "Tier 1"][col].mean()
        tier3_pred = df[df["college"] == "Tier 3"][col].mean()
        college_gap = tier1_pred - tier3_pred
        
        results.append({
            "Technique": name,
            "MSE": mse,
            "R2": r2,
            "Gender Gap ($)": gender_gap,
            "College Gap ($)": college_gap,
            "Gender Gap (%)": (gender_gap / female_pred) * 100 if female_pred != 0 else 0,
            "College Gap (%)": (college_gap / tier3_pred) * 100 if tier3_pred != 0 else 0
        })
        
        print(f"\n{name}:")
        print(f"  MSE: {mse:,.0f}")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  Gender Gap (M-F): ${gender_gap:,.2f} ({(gender_gap/female_pred)*100:.1f}%)")
        print(f"  College Gap (T1-T3): ${college_gap:,.2f} ({(college_gap/tier3_pred)*100:.1f}%)")

# Create comparison DataFrame
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("===== SUMMARY TABLE =====")
print("="*60)
print(results_df.to_string(index=False))

# -----------------------------
# 4️⃣ VISUALIZATION
# -----------------------------
print("\n" + "="*60)
print("4️⃣ GENERATING VISUALIZATIONS")
print("="*60)

# Plot comparison of gender gaps across techniques
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gender Gap Comparison
ax1 = axes[0]
techniques_names = results_df["Technique"].tolist()
gender_gaps = results_df["Gender Gap ($)"].tolist()
colors = ['red' if abs(g) > 20000 else 'orange' if abs(g) > 10000 else 'green' for g in gender_gaps]
bars1 = ax1.barh(techniques_names, gender_gaps, color=colors)
ax1.set_xlabel("Gender Gap (Male - Female) in $")
ax1.set_title("Gender Bias Comparison Across Techniques")
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
for bar, val in zip(bars1, gender_gaps):
    ax1.text(val + 500, bar.get_y() + bar.get_height()/2, f'${val:,.0f}', va='center', fontsize=8)

# College Gap Comparison
ax2 = axes[1]
college_gaps = results_df["College Gap ($)"].tolist()
colors = ['red' if abs(g) > 30000 else 'orange' if abs(g) > 15000 else 'green' for g in college_gaps]
bars2 = ax2.barh(techniques_names, college_gaps, color=colors)
ax2.set_xlabel("College Gap (Tier1 - Tier3) in $")
ax2.set_title("College Tier Bias Comparison Across Techniques")
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
for bar, val in zip(bars2, college_gaps):
    ax2.text(val + 500, bar.get_y() + bar.get_height()/2, f'${val:,.0f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig("bias_mitigation_comparison.png", dpi=150)
plt.show()

# Box plots of predictions by gender for each technique
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for idx, (name, col) in enumerate(techniques.items()):
    if idx < len(axes) and col in df.columns and not df[col].isna().all():
        ax = axes[idx]
        df.boxplot(column=col, by="gender", ax=ax)
        ax.set_title(f"{name}")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Predicted Salary")

# Hide empty subplot if any
for idx in range(len(techniques), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Predicted Salary Distribution by Gender Across Techniques", fontsize=14)
plt.tight_layout()
plt.savefig("predictions_by_gender.png", dpi=150)
plt.show()

# -----------------------------
# Final Outputs
# -----------------------------
print("\n===== SAMPLE PREDICTIONS =====")
sample_cols = ["gender", "college", "salary", "pred_baseline", "pred_undersampled", 
               "pred_oversampled", "pred_smote", "pred_reweighed", "pred_postprocessed"]
print(df[sample_cols].head(10).to_string())

# Save results
output_cols = ["gender", "college", "salary", "pred_baseline", "pred_undersampled", 
               "pred_oversampled", "pred_smote", "pred_combined_under", "pred_reweighed", "pred_postprocessed"]
df[output_cols].to_csv("debiased_salary_predictions_all_techniques.csv", index=False)
results_df.to_csv("bias_mitigation_comparison_results.csv", index=False)

print("\n" + "="*60)
print("✅ OUTPUTS SAVED:")
print("="*60)
print("📄 debiased_salary_predictions_all_techniques.csv")
print("📄 bias_mitigation_comparison_results.csv")
print("📊 bias_mitigation_comparison.png")
print("📊 predictions_by_gender.png")
print("📊 salary_bias_analysis_biased_dataset.png")
