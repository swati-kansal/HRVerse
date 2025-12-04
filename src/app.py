#!/usr/bin/env python3
"""
Flask Backend for Resume Matcher UI
Serves the UI and handles predictions from all ML models
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# Get absolute path to the ui folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_FOLDER = os.path.join(BASE_DIR, 'ui')

app = Flask(__name__, 
            template_folder=UI_FOLDER,
            static_folder=UI_FOLDER)
CORS(app)

# Sample job descriptions (hardcoded for now)
JOB_DESCRIPTIONS = {
    "software_engineer": {
        "title": "Software Engineer",
        "skills": "Python, Java, JavaScript, REST API, SQL, Git, Agile, problem-solving, software development"
    },
    "data_scientist": {
        "title": "Data Scientist",
        "skills": "Python, Machine Learning, TensorFlow, Pandas, SQL, Statistics, Data Analysis, Deep Learning"
    },
    "ux_designer": {
        "title": "UX Designer",
        "skills": "Figma, Adobe XD, User Research, Wireframing, Prototyping, UI Design, Usability Testing"
    },
    "product_manager": {
        "title": "Product Manager",
        "skills": "Product Strategy, Roadmap Planning, Agile, Stakeholder Management, Data Analysis, User Stories"
    },
    "devops_engineer": {
        "title": "DevOps Engineer",
        "skills": "AWS, Docker, Kubernetes, CI/CD, Jenkins, Terraform, Linux, Python, Monitoring"
    }
}


def parse_resume(filename):
    """
    Placeholder function to parse resume PDF.
    Returns hardcoded resume text for now - will be extended with NLP later.
    """
    # Hardcoded resume text for demo purposes
    return """
    Experienced software engineer with 5+ years in Python and Java development.
    Strong background in REST API design, database management with SQL and PostgreSQL.
    Proficient in Git, Agile methodologies, and CI/CD pipelines.
    Experience with cloud services (AWS), Docker containers, and microservices architecture.
    Bachelor's degree in Computer Science. Strong problem-solving and communication skills.
    Previous roles include developing scalable web applications and data processing pipelines.
    """


def predict_with_separate_features(resume_text, job_skills):
    """
    Predict using Logistic Regression with separate features.
    Returns match/non-match prediction.
    """
    # Placeholder - will integrate actual model later
    # For now, return based on simple keyword overlap
    resume_words = set(resume_text.lower().split())
    job_words = set(job_skills.lower().replace(',', ' ').split())
    overlap = len(resume_words.intersection(job_words))
    
    match = overlap >= 3
    confidence = min(overlap * 10, 95)
    
    return {
        "model": "LR - Separate Features",
        "prediction": "Match" if match else "Non-Match",
        "confidence": confidence
    }


def predict_with_combined_features(resume_text, job_skills):
    """
    Predict using Logistic Regression with combined features.
    Returns match/non-match prediction.
    """
    # Placeholder - will integrate actual model later
    resume_words = set(resume_text.lower().split())
    job_words = set(job_skills.lower().replace(',', ' ').split())
    overlap = len(resume_words.intersection(job_words))
    
    match = overlap >= 4
    confidence = min(overlap * 12, 92)
    
    return {
        "model": "LR - Combined Features",
        "prediction": "Match" if match else "Non-Match",
        "confidence": confidence
    }


def predict_with_similarity_features(resume_text, job_skills):
    """
    Predict using Logistic Regression with similarity features.
    Returns match/non-match prediction.
    """
    # Placeholder - will integrate actual model later
    resume_words = set(resume_text.lower().split())
    job_words = set(job_skills.lower().replace(',', ' ').split())
    overlap = len(resume_words.intersection(job_words))
    
    match = overlap >= 2
    confidence = min(overlap * 15, 88)
    
    return {
        "model": "LR - Similarity Features",
        "prediction": "Match" if match else "Non-Match",
        "confidence": confidence
    }


def predict_with_pinecone_llm(resume_text, job_skills):
    """
    Predict using Pinecone + LLM vector similarity.
    Returns match/non-match prediction.
    """
    # Placeholder - will integrate actual Pinecone model later
    resume_words = set(resume_text.lower().split())
    job_words = set(job_skills.lower().replace(',', ' ').split())
    overlap = len(resume_words.intersection(job_words))
    
    match = overlap >= 3
    confidence = min(overlap * 14, 96)
    
    return {
        "model": "Pinecone + LLM",
        "prediction": "Match" if match else "Non-Match",
        "confidence": confidence
    }


@app.route('/')
def index():
    """Serve the main resume matcher page"""
    return render_template('resume_matcher.html')


@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Return list of available jobs"""
    jobs = [{"id": k, "title": v["title"]} for k, v in JOB_DESCRIPTIONS.items()]
    return jsonify(jobs)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Handle prediction request.
    Expects: file (PDF) and job_id
    Returns: predictions from all models
    """
    try:
        # Get uploaded file
        if 'resume' not in request.files:
            return jsonify({"error": "No resume file provided"}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get job ID
        job_id = request.form.get('job_id')
        if not job_id or job_id not in JOB_DESCRIPTIONS:
            return jsonify({"error": "Invalid job selected"}), 400
        
        # Parse resume (placeholder for now)
        resume_text = parse_resume(file.filename)
        
        # Get job skills
        job_data = JOB_DESCRIPTIONS[job_id]
        job_skills = job_data["skills"]
        
        # Run all models
        predictions = [
            predict_with_pinecone_llm(resume_text, job_skills),
            predict_with_separate_features(resume_text, job_skills),
            predict_with_combined_features(resume_text, job_skills),
            predict_with_similarity_features(resume_text, job_skills)
        ]
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "job_title": job_data["title"],
            "predictions": predictions
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Resume Matcher Server...")
    print("📍 Open http://localhost:5001 in your browser")
    app.run(debug=True, port=5001)
