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
        # Common technology terms and job titles that are often misidentified as names
        false_positives = {
            'spring boot', 'spring', 'java', 'python', 'react', 'angular', 'vue',
            'node', 'django', 'flask', 'kubernetes', 'docker', 'aws', 'azure',
            'tensorflow', 'pytorch', 'kafka', 'redis', 'mongodb', 'postgresql',
            'mysql', 'oracle', 'jenkins', 'git', 'linux', 'hibernate', 'rest',
            'soap', 'microservices', 'devops', 'agile', 'scrum', 'jpa',
            'resume', 'cv', 'curriculum', 'vitae', 'objective', 'summary',
            'experience', 'education', 'skills', 'contact', 'competencies',
            # Job titles that might appear after name
            'senior', 'junior', 'lead', 'principal', 'staff', 'associate',
            'engineer', 'developer', 'manager', 'architect', 'analyst',
            'director', 'vp', 'cto', 'ceo', 'engineering', 'software',
            'backend', 'frontend', 'fullstack', 'full-stack', 'data', 'product'
        }
        
        # First try: Look for name patterns at the very beginning (first 200 chars)
        # Names are usually in ALL CAPS or Title Case at the top
        first_part = text[:300]
        
        # Pattern for names in ALL CAPS (like "SWATI KANSAL")
        # Match 2-3 words that are ALL CAPS and likely names (not titles)
        caps_name_match = re.search(r'\b([A-Z]{2,}\s+[A-Z]{2,})\b', first_part)
        if caps_name_match:
            potential_name = caps_name_match.group(1).strip()
            name_words = potential_name.lower().split()
            # Check none of the words are false positives
            if not any(word in false_positives for word in name_words) and len(potential_name) > 3:
                return potential_name.title()  # Convert to Title Case
        
        # Process first 500 chars (name is usually at the top)
        doc = self.nlp(text[:500])
        
        # Look for PERSON entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                name_words = name.lower().split()
                # Filter out false positives
                if (len(name) > 2 and 
                    not any(word in false_positives for word in name_words)):
                    return name
        
        # Fallback: Try to find name from first line
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            words = first_line.split()
            # Take only first 2-3 words that look like a name
            name_words = []
            for word in words[:3]:
                if word.isalpha() and word.lower() not in false_positives:
                    name_words.append(word)
                else:
                    break
            if len(name_words) >= 2:
                return ' '.join(name_words)
        
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
        text_lower = text.lower()
        
        # Direct patterns for "X years experience"
        direct_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*(?:in\s*)?(?:the\s*)?(?:industry|field)',
            r'(\d+)\+?\s*years?\s*(?:in|of|working)',
            r'(\d+)\+?\s*years?\s*(?:as\s*a?\s*)?(?:developer|engineer|designer|manager)',
            r'(?:over|more\s*than)\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?\b',
        ]
        
        for pattern in direct_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        
        # Calculate from work history dates
        # Look for patterns like "2011 – Present", "Jan 2020 - Dec 2022", etc.
        from datetime import datetime
        current_year = datetime.now().year
        
        # Find all years mentioned in work experience context
        year_pattern = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?\s*(\d{4})\s*[-–—]\s*(?:present|current|now|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)?\s*(\d{4}))'
        matches = re.findall(year_pattern, text_lower)
        
        if matches:
            earliest_year = current_year
            for match in matches:
                start_year = int(match[0])
                if 1990 <= start_year <= current_year:
                    earliest_year = min(earliest_year, start_year)
            
            if earliest_year < current_year:
                return current_year - earliest_year
        
        return None
    
    def extract_education(self, text: str) -> list:
        """
        Extract education information from resume text.
        This function extracts any education-related content found in the resume,
        including degrees, certifications, universities, and fields of study.
        """
        text_lower = text.lower()
        found_education = []
        
        # Common degree patterns with variations
        degree_patterns = [
            # B.Tech variations
            (r'\bb\.?\s*-?\s*tech\b', 'B.Tech'),
            (r'\bbtech\b', 'B.Tech'),
            # M.Tech variations
            (r'\bm\.?\s*-?\s*tech\b', 'M.Tech'),
            (r'\bmtech\b', 'M.Tech'),
            # B.E variations
            (r'\bb\.?\s*e\.?\b(?!\s*fore)', 'B.E'),  # Avoid "before"
            # M.E variations
            (r'\bm\.?\s*e\.?\b', 'M.E'),
            # B.Sc variations
            (r'\bb\.?\s*-?\s*sc\.?\b', 'B.Sc'),
            (r'\bbsc\b', 'B.Sc'),
            # M.Sc variations
            (r'\bm\.?\s*-?\s*sc\.?\b', 'M.Sc'),
            (r'\bmsc\b', 'M.Sc'),
            # Bachelor's degrees
            (r"\bbachelor'?s?\s*(?:of\s*)?\b(?:science|engineering|arts|technology|commerce|computer\s*(?:science|applications))", 'Bachelors'),
            (r'\bbachelor\s*degree\b', 'Bachelors Degree'),
            # Master's degrees
            (r"\bmaster'?s?\s*(?:of\s*)?\b(?:science|engineering|arts|technology|business\s*administration|computer\s*(?:science|applications))", 'Masters'),
            (r'\bmaster\s*degree\b', 'Masters Degree'),
            # PhD / Doctorate
            (r'\bph\.?\s*d\.?\b', 'PhD'),
            (r'\bdoctorate\b', 'Doctorate'),
            (r'\bdoctor\s*of\b', 'Doctorate'),
            # MBA
            (r'\bm\.?\s*b\.?\s*a\.?\b', 'MBA'),
            # BBA
            (r'\bb\.?\s*b\.?\s*a\.?\b', 'BBA'),
            # BCA / MCA
            (r'\bb\.?\s*c\.?\s*a\.?\b', 'BCA'),
            (r'\bm\.?\s*c\.?\s*a\.?\b', 'MCA'),
            # B.Com / M.Com
            (r'\bb\.?\s*-?\s*com\.?\b', 'B.Com'),
            (r'\bm\.?\s*-?\s*com\.?\b', 'M.Com'),
            (r'\bbcom\b', 'B.Com'),
            (r'\bmcom\b', 'M.Com'),
            # B.A / M.A
            (r'\bb\.?\s*a\.?\b(?!\s*(?:ck|se|r|d|ll|n|g|t))', 'B.A'),  # Avoid "back", "base", etc.
            (r'\bm\.?\s*a\.?\b(?!\s*(?:ke|in|na|ny|x|ch|jo|tt))', 'M.A'),  # Avoid "make", "main", etc.
            # Diploma
            (r'\bdiploma\b', 'Diploma'),
            (r'\bpg\s*diploma\b', 'PG Diploma'),
            (r'\bpost\s*graduate\s*diploma\b', 'PG Diploma'),
            # High School
            (r'\bhigh\s*school\b', 'High School'),
            (r'\b12th\b', '12th Grade'),
            (r'\b10th\b', '10th Grade'),
            (r'\bclass\s*(?:12|xii)\b', '12th Grade'),
            (r'\bclass\s*(?:10|x)\b', '10th Grade'),
            # Certifications
            (r'\bcertificat(?:e|ion)\b', 'Certification'),
            # Associate degree
            (r'\bassociate\s*degree\b', 'Associate Degree'),
            (r'\ba\.?\s*s\.?\b.*\bdegree\b', 'Associate Degree'),
        ]
        
        # First pass: Look for specific degree patterns
        for pattern, label in degree_patterns:
            if re.search(pattern, text_lower):
                found_education.append(label)
        
        # Second pass: Look for education section and extract more context
        # Find education section in the resume
        education_section_patterns = [
            r'education\s*[:\-]?\s*(.*?)(?:experience|skills|projects|work|employment|$)',
            r'academic\s*(?:background|qualifications?)\s*[:\-]?\s*(.*?)(?:experience|skills|projects|work|employment|$)',
            r'qualifications?\s*[:\-]?\s*(.*?)(?:experience|skills|projects|work|employment|$)',
        ]
        
        # Look for university/college names
        institution_patterns = [
            r'\buniversity\b',
            r'\bcollege\b', 
            r'\binstitute\b',
            r'\bschool\s*of\b',
            r'\biit\b',
            r'\bnit\b',
            r'\bits\b',
        ]
        
        for pattern in institution_patterns:
            if re.search(pattern, text_lower):
                # Check if we haven't already added a more specific degree
                if not any(deg in ['Bachelors', 'Masters', 'PhD', 'B.Tech', 'M.Tech'] for deg in found_education):
                    # Try to extract what type of education from context
                    context_match = re.search(pattern + r'.*?(\d{4})', text_lower)
                    if context_match and 'University Education' not in found_education:
                        found_education.append('University Education')
                        break
        
        # Look for fields of study
        field_patterns = [
            (r'computer\s*science', 'Computer Science'),
            (r'information\s*technology', 'Information Technology'),
            (r'software\s*engineering', 'Software Engineering'),
            (r'electrical\s*engineering', 'Electrical Engineering'),
            (r'mechanical\s*engineering', 'Mechanical Engineering'),
            (r'civil\s*engineering', 'Civil Engineering'),
            (r'electronics?\s*(?:and\s*)?communications?', 'Electronics & Communication'),
            (r'data\s*science', 'Data Science'),
            (r'business\s*administration', 'Business Administration'),
            (r'economics', 'Economics'),
            (r'mathematics', 'Mathematics'),
            (r'physics', 'Physics'),
            (r'chemistry', 'Chemistry'),
        ]
        
        for pattern, label in field_patterns:
            if re.search(pattern, text_lower):
                if label not in found_education:
                    found_education.append(label)
        
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
