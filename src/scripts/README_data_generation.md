# Synthetic Data Generation for Resume-Job Matching

This script generates synthetic datasets for training and evaluating resume-job skill matching models using OpenAI's GPT API.

## Features

- **Batch Processing**: Generates data in configurable batches to respect API rate limits
- **Diverse Bias Types**: Includes 8 categories of controlled biases for robust model training
- **Realistic Data**: Creates authentic-looking resumes and job descriptions with proper formatting
- **Balanced Dataset**: Ensures ~50% positive and ~50% negative matching cases
- **Progress Tracking**: Saves progress and provides detailed logging
- **Error Handling**: Robust retry logic and graceful error recovery

## Generated Data Format

The script generates a CSV file with these columns:
- `name`: Diverse, realistic person names from global cultures
- `resume_text`: Formatted resume content with various styles and biases
- `job_skill_text`: Realistic job requirements and skill expectations
- `match_label`: Binary classification (1=match, 0=no match)
- `bias_type`: Type of controlled bias injected into the record

## Bias Categories

The generator includes these controlled biases:

1. **Textual Biases**: Synonyms, abbreviations, grammar variations, spelling errors
2. **Structural Biases**: Missing sections, inconsistent formatting, scattered skills
3. **Skill-Match Biases**: Partial overlaps, domain mismatches, seniority gaps
4. **Experience Biases**: Outdated skills, legacy tools, experience vs. skill mismatch
5. **Fairness Biases**: Cultural diversity, geographic variation (evaluation only)
6. **Domain-Specific Biases**: Resume padding, buzzwords, mixed experience
7. **Missing/Noisy Data**: Incomplete information, HTML artifacts, duplicates
8. **Human-Error Biases**: Typos, copy-paste errors, formatting inconsistencies

## Setup

1. **Install Dependencies**:
   ```bash
   cd /path/to/ai-hiring-portal
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY='your-openai-api-key-here'
   ```
   
   Or add to your shell profile:
   ```bash
   echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.zshrc
   source ~/.zshrc
   ```

## Usage

### Option 1: Interactive Runner (Recommended)
```bash
cd src/scripts
python run_data_generation.py
```

### Option 2: Direct Script Usage
```bash
cd src/scripts
python data_generation.py
```

### Option 3: Programmatic Usage
```python
from src.scripts.data_generation import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator(
    api_key="your-openai-key",
    batch_size=25  # Records per API call
)

# Generate dataset
output_file = generator.generate_dataset(
    total_records=1000,
    output_file="custom_output.csv"  # Optional
)
```

## Configuration Options

- **total_records**: Number of records to generate (default: 1000)
- **batch_size**: Records per API call (default: 25, max recommended: 50)
- **output_file**: Custom output path (default: auto-generated in storage/synthetic_data/)

## Output

The script creates:
- **Main CSV file**: Complete synthetic dataset
- **Progress logging**: Detailed generation progress and statistics
- **Error handling**: Automatic retries and graceful failure recovery

Example output location:
```
storage/synthetic_data/synthetic_resume_job_matching_1000_records.csv
```

## Performance & Costs

- **Generation Speed**: ~25 records per API call (1-2 seconds per batch)
- **Estimated Cost**: ~$0.10-0.20 per 1000 records (using GPT-4o-mini)
- **Rate Limits**: Built-in delays to respect OpenAI rate limits
- **Memory Usage**: Minimal - processes data in batches

## Example Generated Data

```csv
name,resume_text,job_skill_text,match_label,bias_type
"Sarah Chen","Software Engineer with 5+ years experience in Python, React, and AWS...","Senior Developer: Python, JavaScript, Cloud platforms required...","1","textual_bias_synonyms"
"Ahmed Hassan","Marketing specialist with social media expertize and campagin management...","Data Scientist: Machine learning, statistics, Python required...","0","human_error_typos"
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure OPENAI_API_KEY is set correctly
2. **Rate Limits**: Reduce batch_size if hitting rate limits
3. **Memory Issues**: Use smaller batch sizes for large datasets
4. **Network Timeouts**: Script includes automatic retry logic

### Quality Control

- Review generated data for realism and bias distribution
- Check match_label balance (should be ~50/50)
- Validate bias_type variety across records
- Ensure text lengths are realistic for resumes and job descriptions

## Integration

The generated data can be used with:
- Machine learning training pipelines
- Model evaluation and testing
- Bias detection and fairness analysis
- Resume parsing and matching systems

## Next Steps

1. **Data Quality Review**: Examine samples for realism and diversity
2. **Model Training**: Use with scikit-learn, transformers, or other ML frameworks  
3. **Bias Analysis**: Evaluate model performance across different bias types
4. **Production Integration**: Incorporate into your hiring portal ML pipeline
