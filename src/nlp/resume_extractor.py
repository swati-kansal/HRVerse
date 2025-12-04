#!/usr/bin/env python3
"""
Resume NLP Extractor
Extracts structured information from PDF resumes using NLP
"""

import re
import os
from typing import Dict, Optional
from PyPDF2 import PdfReader
import spacy

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ Downloading spaCy model 'en_core_web_sm'...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class ResumeExtractor:
    """Extract structured data from PDF resumes"""
    
    def __init__(self):
        self.nlp = nlp
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract raw text from a PDF file.
        
        Args:
            pdf_file: File object or path to PDF
            
        Returns:
            Extracted text as string
        """
        try:
            reader = PdfReader(pdf_file)
            text_parts = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            full_text = "\n".join(text_parts)
            # Clean up whitespace
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            return full_text
            
        except Exception as e:
            print(f"❌ Error extracting PDF text: {e}")
            return ""
    
    def extract_name(self, text: str) -> str:
        """
        Extract candidate name from resume text using NLP.
        
        Args:
            text: Resume text
            
        Returns:
            Extracted name or "Unknown"
        """
        # Process first 500 chars (name is usually at the top)
        doc = self.nlp(text[:500])
        
        # Look for PERSON entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Return the first person name found
                name = ent.text.strip()
                # Filter out common false positives
                if len(name) > 2 and not name.lower() in ['resume', 'cv', 'curriculum']:
                    return name
        
        # Fallback: Try to find name from first line
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if first line looks like a name (2-4 words, no special chars)
            words = first_line.split()
            if 1 <= len(words) <= 4 and all(w.isalpha() for w in words):
                return first_line
        
        return "Unknown Candidate"
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address from resume text."""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(email_pattern, text)
        return match.group(0) if match else None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from resume text."""
        # Match various phone formats
        phone_patterns = [
            r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+?\d{10,12}',
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def extract_skills(self, text: str) -> list:
        """
        Extract skills from resume text.
        
        Args:
            text: Resume text
            
        Returns:
            List of extracted skills
        """
        # Common technical skills to look for
        skill_keywords = [
            # Programming Languages
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift',
            'kotlin', 'go', 'rust', 'scala', 'r', 'matlab', 'sql', 'typescript',
            # Frameworks & Libraries
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js',
            'express', 'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy',
            # Tools & Platforms
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
            'linux', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
            # Concepts
            'machine learning', 'deep learning', 'ai', 'data analysis', 'agile',
            'scrum', 'rest api', 'microservices', 'devops', 'ci/cd',
            # Design
            'figma', 'adobe', 'photoshop', 'illustrator', 'ux', 'ui', 'sketch',
            # Soft Skills
            'leadership', 'communication', 'teamwork', 'problem-solving',
            'project management', 'analytical'
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in skill_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        return list(set(found_skills))
    
    def extract_experience_years(self, text: str) -> Optional[int]:
        """Extract years of experience from resume text."""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*(?:in\s*)?(?:the\s*)?(?:industry|field)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def extract_education(self, text: str) -> list:
        """Extract education information from resume text."""
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'mba', 'b.tech', 'm.tech',
            'b.sc', 'm.sc', 'b.e', 'm.e', 'bca', 'mca', 'diploma'
        ]
        
        text_lower = text.lower()
        found_education = []
        
        for edu in education_keywords:
            if edu in text_lower:
                found_education.append(edu.upper())
        
        return list(set(found_education))
    
    def extract_all(self, pdf_file) -> Dict:
        """
        Extract all information from a PDF resume.
        
        Args:
            pdf_file: File object or path to PDF
            
        Returns:
            Dictionary with extracted information
        """
        # Extract raw text
        resume_text = self.extract_text_from_pdf(pdf_file)
        
        if not resume_text:
            return {
                "success": False,
                "error": "Could not extract text from PDF",
                "name": "Unknown",
                "resume_text": "",
                "email": None,
                "phone": None,
                "skills": [],
                "experience_years": None,
                "education": []
            }
        
        # Extract structured information
        return {
            "success": True,
            "name": self.extract_name(resume_text),
            "resume_text": resume_text,
            "email": self.extract_email(resume_text),
            "phone": self.extract_phone(resume_text),
            "skills": self.extract_skills(resume_text),
            "experience_years": self.extract_experience_years(resume_text),
            "education": self.extract_education(resume_text)
        }


# Singleton instance
_extractor = None

def get_extractor() -> ResumeExtractor:
    """Get or create the resume extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = ResumeExtractor()
    return _extractor


def extract_resume_data(pdf_file) -> Dict:
    """
    Convenience function to extract data from a PDF resume.
    
    Args:
        pdf_file: File object or path to PDF
        
    Returns:
        Dictionary with extracted information
    """
    extractor = get_extractor()
    return extractor.extract_all(pdf_file)


if __name__ == "__main__":
    # Test with a sample PDF
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"📄 Extracting data from: {pdf_path}")
        
        data = extract_resume_data(pdf_path)
        
        print("\n📋 Extracted Information:")
        print(f"   Name: {data['name']}")
        print(f"   Email: {data['email']}")
        print(f"   Phone: {data['phone']}")
        print(f"   Experience: {data['experience_years']} years")
        print(f"   Education: {', '.join(data['education'])}")
        print(f"   Skills: {', '.join(data['skills'][:10])}...")
        print(f"\n📝 Resume Text Preview:")
        print(f"   {data['resume_text'][:300]}...")
    else:
        print("Usage: python resume_extractor.py <path_to_pdf>")
