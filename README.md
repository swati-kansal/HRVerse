# 🤖 AI Hiring Portal (IIM Capstone Project)

## 📌 Project Overview
The **AI Hiring Portal** is a recruitment automation system designed to speed up resume screening and improve hiring accuracy.  
It parses resumes, compares them with job descriptions (JDs), generates **match scores**, highlights **missing skills**, and helps recruiters shortlist candidates quickly.  

This project will demonstrate:  
1. **Classroom Models (Baseline)** – ML models taught in our IIM AI/ML course.  
2. **External Models (Improved)** – Advanced real-world AI approaches for higher accuracy.  

---

## 🎯 Problem Statement
Recruiters spend hours reading and shortlisting resumes. This is **time-consuming, error-prone, and biased**.  
Our portal automates the process using **AI/ML** to:  
- Parse resumes (extract skills, education, experience).  
- Match resumes with JDs.  
- Classify candidates as **MATCHED / NEEDS REVIEW / REJECTED**.  
- Highlight missing skills for feedback.  

---

pip install -r requirements.txt after clone the Application

## 📂 Project Structure

ai-hiring-portal/
│
├── README.md                # Project overview & setup instructions
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignore dataset, env files, cache
│
├── docs/                    # Documentation & PPTs
│   ├── Proposal.md
│   ├── Technical_Flow.md
│   ├── Results_Report.md
│   └── Review_PPT.pptx
│
├── dataset/                 # Resumes & JDs (shared folder, not in GitHub)
│   ├── resumes/
│   └── jds/
│
├── notebooks/               # Jupyter Notebooks
│   ├── 01_baseline_models.ipynb
│   ├── 02_improved_models.ipynb
│   └── 03_comparison.ipynb
│
├── src/                     # Source code
│   ├── preprocessing/       # Resume & JD cleaning
│   │   ├── resume_parser.py
│   │   ├── jd_parser.py
│   │   └── text_cleaner.py
│   │
│   ├── models/              # ML models
│   │   ├── baseline_models.py    # TF-IDF, Logistic, DecisionTree, NaiveBayes, KMeans
│   │   ├── improved_models.py    # BERT, RandomForest, XGBoost
│   │   └── evaluation.py         # accuracy, precision, recall, F1
│   │
│   ├── storage/             # Database & file handling
│   │   ├── sqlite_handler.py
│   │   └── file_manager.py
│   │
│   ├── api/                 # Backend (FastAPI/Flask)
│   │   ├── main.py
│   │   └── routes.py
│   │
│   └── ui/                  # Frontend UI
│       ├── login.html
│       ├── recruiter.html
│       ├── candidate.html
│       ├── admin.html
│       ├── style.css
│       └── js/
│           ├── login.js
│           ├── recruiter.js
│           └── candidate.js
│
├── tests/                   # Unit tests
│   ├── test_parser.py
│   ├── test_baseline.py
│   └── test_improved.py




---

## 🛠 Pre-Installation Steps

1. **Install Python 3.9+**  
   [Download Python](https://www.python.org/downloads/)  

2. **Install Git**  
   [Download Git](https://git-scm.com/downloads)  

3. **Clone Repo (after it’s created on GitHub)**  
   ```bash
   git clone https://github.com/<your-username>/ai-hiring-portal.git
   cd ai-hiring-portal


📚 Models to Implement
📖 Classroom Models (Baseline – Covered in Course)

1.TF-IDF + Cosine Similarity → Basic JD–Resume text similarity.
2.Logistic Regression → Classify resumes as MATCHED / REJECTED.
3.Decision Tree → Flowchart-based classification.
4.Naïve Bayes → Probability-based matching.
5.K-Means Clustering → Group candidates by skills.
6.Purpose: Simple, explainable, course-aligned models.

🚀 External Models (Improved – Real-World Accuracy)

1Sentence Transformers (BERT embeddings) → Understands meaning, not just keywords.

2.Random Forest / XGBoost → Stronger classifiers, ensemble approach.

3.Skill Gap Detection (NER) → Highlights missing skills in resumes.
