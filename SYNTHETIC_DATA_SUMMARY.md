# Synthetic Data Generation System - Implementation Summary

## What I've Created

I've built a comprehensive synthetic data generation system for your AI hiring portal that uses OpenAI's API to create realistic resume-job matching datasets.

## Files Created

### 1. `data_generation.py` - Main Generation Script
**Features:**
- **Batch Processing**: Generates data in configurable batches (default: 25 records per batch)
- **8 Bias Categories**: Implements all required bias types for robust ML training
- **Error Handling**: Automatic retries, progress saving, and graceful failure recovery
- **Statistics**: Detailed logging and dataset quality analysis
- **Flexible Output**: Customizable output locations and file naming

### 2. `run_data_generation.py` - User-Friendly Runner
**Features:**
- Interactive command-line interface
- Parameter configuration (records count, batch size, output path)
- API key validation and setup guidance
- Progress monitoring and success confirmation

### 3. `test_data_generation.py` - Verification Script
**Features:**
- Quick 10-record test generation
- Data quality validation
- System functionality verification
- Cleanup and error reporting

### 4. `README_data_generation.md` - Complete Documentation
**Includes:**
- Setup instructions
- Usage examples
- Configuration options
- Troubleshooting guide
- Performance and cost estimates

## Key Features

### Bias Implementation
The system generates data with these controlled biases:

1. **Textual**: Synonyms, abbreviations, spelling errors
2. **Structural**: Missing sections, inconsistent formatting
3. **Skill-Match**: Partial overlaps, domain mismatches, seniority gaps
4. **Experience**: Outdated skills, legacy tools, experience mismatches
5. **Fairness**: Cultural/geographic diversity (evaluation only)
6. **Domain-Specific**: Resume padding, buzzwords, mixed experience
7. **Missing/Noisy**: Incomplete data, HTML artifacts, duplicates
8. **Human-Error**: Typos, copy-paste errors, formatting issues

### Data Quality
- **Balanced Labels**: ~50% positive, ~50% negative matches
- **Realistic Content**: Authentic resume and job description formats
- **Global Diversity**: Names and backgrounds from various cultures
- **Variable Length**: Short and long resumes/job descriptions
- **Professional Domains**: Multiple industries and skill sets

## Usage Instructions

### Quick Start
```bash
# 1. Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# 2. Navigate to scripts directory
cd src/scripts

# 3. Run the interactive generator
python run_data_generation.py

# OR run a quick test first
python test_data_generation.py
```

### Programmatic Usage
```python
from src.scripts.data_generation import SyntheticDataGenerator

generator = SyntheticDataGenerator(batch_size=25)
output_file = generator.generate_dataset(total_records=1000)
```

## Output Format

Generated CSV with columns:
- `name`: Realistic global names
- `resume_text`: Formatted resume content with biases
- `job_skill_text`: Realistic job requirements
- `match_label`: Binary (1=match, 0=no match)  
- `bias_type`: Specific bias category applied

## Performance & Costs

- **Speed**: ~25 records per API call (1-2 seconds per batch)
- **Cost**: ~$0.10-0.20 per 1000 records (GPT-4o-mini)
- **Scalability**: Handles 1000+ records efficiently with progress tracking
- **Rate Limits**: Built-in delays respect OpenAI API limits

## Next Steps

1. **Set API Key**: Configure your OpenAI API key
2. **Test System**: Run `test_data_generation.py` first
3. **Generate Data**: Use `run_data_generation.py` for full dataset
4. **Review Quality**: Examine generated samples for realism
5. **Train Models**: Use data with your ML pipelines

## Integration Points

This system integrates with:
- Your existing ML pipeline in `src/scripts/`
- Resume storage system in `src/storage/`
- Model training scripts for classification tasks
- Bias evaluation and fairness testing workflows

The generated synthetic data will provide a robust foundation for training your resume-job matching models with controlled biases and realistic variety.
