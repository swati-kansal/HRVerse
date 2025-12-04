#!/usr/bin/env python3
"""
Synthetic Resume-Job Matching Data Generator

This script generates synthetic data for resume-job skill matching classification
using OpenAI's API. It creates batches of diverse, realistic data with various
controlled biases for training and evaluation purposes.
"""

import openai
import csv
import os
import json
import time
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    def __init__(self, api_key: Optional[str] = None, batch_size: int = 50):
        """
        Initialize the data generator.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            batch_size: Number of records to generate per API call
        """
        self.client = openai.OpenAI(
            api_key="""
        )
        self.batch_size = batch_size
        
    def create_prompt(self, num_records: int) -> str:
        """Create the prompt for generating synthetic data."""
        return f"""Generate a synthetic dataset for a Resume–Job Skill Match classification task.

Output Format:
- Return the dataset as a CSV table.
- Columns must be in this exact order:
  name, resume_text, job_skill_text, match_label, bias_type

Record Requirements:
- Generate {num_records} rows.
- "name" must be a realistic person name (diverse global names).
- "resume_text" must look like real resumes with variation in format and content.
- "job_skill_text" must list realistic job skills, responsibilities, and expectations.
- "match_label":
    1 = resume skills realistically match job skills (≥70% overlap)
    0 = skills do NOT match or candidate is under-/over-qualified
- Ensure approximately 50% positive and 50% negative cases.
- "bias_type" should specify which intentional bias was injected.

Include a wide variety of controlled biases in the dataset:
1. Textual biases:
   - synonyms, abbreviations, grammar variations
   - spelling mistakes, noisy text, long/short resume formats
2. Structural biases:
   - missing sections, inconsistent bullet styles, skills in experience section
3. Skill-match biases:
   - partial overlaps, strong matches, domain mismatches, mismatched seniority
4. Experience biases:
   - outdated skills, old tools, low experience despite skill match, recency issues
5. Fairness biases for evaluation only (should NOT influence match_label):
   - different cultures, names, universities, and locations
6. Domain-specific biases:
   - resume padding, buzzword stuffing, mixed domain experience
7. Missing/noisy data:
   - missing skills, incomplete job descriptions, HTML artifacts, duplicated lines
8. Human-error biases:
   - typos, mis-typed company names, copy-paste issues

Output ONLY the CSV content, without explanation or formatting outside the CSV.

Example CSV header:
name,resume_text,job_skill_text,match_label,bias_type"""

    def generate_batch(self, batch_num: int, records_in_batch: int) -> List[Dict]:
        """
        Generate a batch of synthetic data using OpenAI API.
        
        Args:
            batch_num: Current batch number (for logging)
            records_in_batch: Number of records to generate in this batch
            
        Returns:
            List of dictionaries containing the generated data
        """
        logger.info(f"Generating batch {batch_num} with {records_in_batch} records...")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using cost-effective model for data generation
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert data scientist creating high-quality synthetic training data for machine learning models. Generate realistic, diverse data with controlled biases."
                    },
                    {
                        "role": "user", 
                        "content": self.create_prompt(records_in_batch)
                    }
                ],
                temperature=0.8,  # Higher temperature for more variety
                max_tokens=4000,  # Adjust based on batch size
            )
            
            # Extract CSV content from response
            if response.choices[0].message.content is None:
                raise ValueError("Empty response from OpenAI API")
            csv_content = response.choices[0].message.content.strip()
            
            # Parse CSV content
            csv_lines = csv_content.split('\n')
            
            # Find header line and data lines
            header_found = False
            data_lines = []
            
            for line in csv_lines:
                if not header_found and 'name,resume_text,job_skill_text,match_label,bias_type' in line:
                    header_found = True
                    continue
                elif header_found and line.strip():
                    data_lines.append(line)
            
            # Parse data lines into dictionaries
            records = []
            reader = csv.DictReader(data_lines, fieldnames=['name', 'resume_text', 'job_skill_text', 'match_label', 'bias_type'])
            
            for row in reader:
                if all(row.values()):  # Skip empty rows
                    # Clean and validate the data
                    record = {
                        'name': row['name'].strip().strip('"'),
                        'resume_text': row['resume_text'].strip().strip('"'),
                        'job_skill_text': row['job_skill_text'].strip().strip('"'),
                        'match_label': int(row['match_label'].strip()),
                        'bias_type': row['bias_type'].strip().strip('"')
                    }
                    records.append(record)
            
            logger.info(f"Successfully generated {len(records)} records in batch {batch_num}")
            return records
            
        except Exception as e:
            logger.error(f"Error generating batch {batch_num}: {str(e)}")
            return []

    def generate_dataset(self, total_records: int = 1000, output_file: Optional[str] = None) -> str:
        """
        Generate the complete synthetic dataset.
        
        Args:
            total_records: Total number of records to generate
            output_file: Path to output CSV file
            
        Returns:
            Path to the generated CSV file
        """
        if output_file is None:
            # Create output directory if it doesn't exist
            output_dir = Path("../../storage/synthetic_data")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(output_dir / f"synthetic_resume_job_matching_{total_records}_records.csv")
        
        # Calculate number of batches
        num_batches = (total_records + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Starting generation of {total_records} records in {num_batches} batches...")
        
        all_records = []
        
        for batch_num in range(1, num_batches + 1):
            # Calculate records for this batch
            records_remaining = total_records - len(all_records)
            records_in_batch = min(self.batch_size, records_remaining)
            
            if records_in_batch <= 0:
                break
                
            # Generate batch with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                batch_records = self.generate_batch(batch_num, records_in_batch)
                
                if batch_records:
                    all_records.extend(batch_records)
                    break
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retrying batch {batch_num} (attempt {attempt + 2}/{max_retries})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"Failed to generate batch {batch_num} after {max_retries} attempts")
            
            # Add delay between batches to respect rate limits
            if batch_num < num_batches:
                time.sleep(1)
            
            # Save progress periodically
            if len(all_records) % 200 == 0:
                self._save_progress(all_records, f"{output_file}.tmp")
        
        # Save final dataset
        if output_file is not None:
            self._save_to_csv(all_records, output_file)
        
        # Clean up temporary file
        temp_file = f"{output_file}.tmp"
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        logger.info(f"Dataset generation complete! Generated {len(all_records)} records.")
        logger.info(f"Output saved to: {output_file}")
        
        # Print dataset statistics
        self._print_statistics(all_records)
        
        return output_file or ""

    def _save_progress(self, records: List[Dict], filename: str):
        """Save progress to a temporary file."""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                if records:
                    writer = csv.DictWriter(f, fieldnames=['name', 'resume_text', 'job_skill_text', 'match_label', 'bias_type'])
                    writer.writeheader()
                    writer.writerows(records)
            logger.info(f"Progress saved: {len(records)} records")
        except Exception as e:
            logger.error(f"Error saving progress: {str(e)}")

    def _save_to_csv(self, records: List[Dict], filename: str):
        """Save records to CSV file."""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                if records:
                    writer = csv.DictWriter(f, fieldnames=['name', 'resume_text', 'job_skill_text', 'match_label', 'bias_type'])
                    writer.writeheader()
                    writer.writerows(records)
            logger.info(f"Dataset saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")

    def _print_statistics(self, records: List[Dict]):
        """Print dataset statistics."""
        if not records:
            return
            
        df = pd.DataFrame(records)
        
        logger.info("=== Dataset Statistics ===")
        logger.info(f"Total records: {len(records)}")
        
        # Match label distribution
        match_dist = df['match_label'].value_counts()
        logger.info(f"Match distribution: {dict(match_dist)}")
        logger.info(f"Match balance: {match_dist[1]}/{match_dist[0]} (positive/negative)")
        
        # Bias type distribution
        bias_dist = df['bias_type'].value_counts()
        logger.info(f"Top bias types: {dict(bias_dist.head(10))}")
        
        # Text length statistics
        resume_lengths = df['resume_text'].str.len()
        job_lengths = df['job_skill_text'].str.len()
        
        logger.info(f"Resume text length - Mean: {resume_lengths.mean():.0f}, Std: {resume_lengths.std():.0f}")
        logger.info(f"Job text length - Mean: {job_lengths.mean():.0f}, Std: {job_lengths.std():.0f}")


def main():
    """Main function to run the data generation."""
    # Use hardcoded API key or environment variable
    api_key = "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"
    if not api_key:
        logger.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return
    
    # Initialize generator
    generator = SyntheticDataGenerator(api_key=api_key, batch_size=25)  # Smaller batches for better quality
    
    # Generate dataset
    try:
        output_file = generator.generate_dataset(total_records=1000)
        print(f"\n✅ Dataset generation completed successfully!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Open the file to examine the generated synthetic data.")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()