import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("src/storage/salary_prediction_dataset_biased.csv")

# -----------------------------
# 1️⃣ BIAS DETECTION
# -----------------------------

print("===== BASIC STATS =====")
print(df.groupby("gender")["salary"].mean())
print(df.groupby("college")["salary"].mean())

# -----------------------------
# Gender Salary Gap
# -----------------------------
gender_salary = df.groupby("gender")["salary"].mean()
print("\n===== GENDER BIAS =====")
print(gender_salary)
print("Gap (Male - Female): ", gender_salary["Male"] - gender_salary["Female"])

# T-test Gender
male_salary = df[df.gender == "Male"].salary
female_salary = df[df.gender == "Female"].salary
t_stat, p_val = ttest_ind(male_salary, female_salary, equal_var=False)
print("T-test p-value:", p_val)

# -----------------------------
# College Tier Bias
# -----------------------------
tier_salary = df.groupby("college")["salary"].mean()
print("\n===== COLLEGE TIER BIAS =====")
print(tier_salary)

# -----------------------------
# Correlation with Salary
# -----------------------------
df_encoded = pd.get_dummies(df, drop_first=True)
corr = df_encoded.corr()["salary"].sort_values(ascending=False)
print("\n===== CORRELATION WITH SALARY =====")
print(corr)

# -----------------------------
# Visual Bias Detection
# -----------------------------
sns.boxplot(x="gender", y="salary", data=df)
plt.title("Salary Distribution by Gender")
plt.show()

sns.boxplot(x="college", y="salary", data=df)
plt.title("Salary Distribution by College Tier")
plt.show()

# -----------------------------
# 2️⃣ BIAS MITIGATION PIPELINE
# -----------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Remove sensitive attributes from training
X = df.drop(columns=["salary", "gender", "college"])
X = pd.get_dummies(X)

y = df["salary"]

# -----------------------------
# Technique 1: Preprocessing – Reweighing
# -----------------------------
def compute_weights(df, sensitive_attr):
    probs = df[sensitive_attr].value_counts(normalize=True)
    weights = df[sensitive_attr].map(lambda x: 1 / probs[x])
    weights = weights / weights.mean()
    return weights

df["gender_weight"] = compute_weights(df, "gender")
df["college_weight"] = compute_weights(df, "college")

df["final_weight"] = (df["gender_weight"] + df["college_weight"]) / 2

# -----------------------------
# Technique 2: In-processing Model
# Train model with fairness weights
# -----------------------------
fair_model = LinearRegression()

fair_model.fit(X, y, sample_weight=df["final_weight"])

# -----------------------------
# Technique 3: Post-processing
# Remove prediction bias
# -----------------------------
def debias_salary(pred, gender, college):
    adj = 0
    if gender == "Female":
        adj += pred * 0.10
    if college == "Tier 3":
        adj += pred * 0.10
    return pred + adj

df["pred_salary_raw"] = fair_model.predict(X)
df["pred_salary_debiased"] = df.apply(
    lambda r: debias_salary(r["pred_salary_raw"], r["gender"], r["college"]),
    axis=1
)

# -----------------------------
# Final Outputs
# -----------------------------
print("\n===== RAW MODEL PREDICTION (biased) =====")
print(df["pred_salary_raw"].head())

print("\n===== DE-BIASED MODEL PREDICTION =====")
print(df["pred_salary_debiased"].head())

df.to_csv("debiased_salary_predictions.csv", index=False)

print("\nSaved debiased predictions to 'debiased_salary_predictions.csv'")
