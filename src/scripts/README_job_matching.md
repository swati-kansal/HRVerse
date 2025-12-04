# Job Matching with Pinecone Integration

This module provides cosine similarity-based job matching functionality that searches for the top matching candidates based on job description details.

## Features

- 🔍 **Cosine Similarity Search**: Find candidates using semantic similarity
- 📊 **Match Percentage Scoring**: Convert cosine similarity to percentage scores
- 🎯 **Top-K Results**: Get top 3 (or custom number) best matches
- 📝 **Comprehensive Job Description**: Support for job title, keywords, requirements, and descriptions
- 🚀 **Multiple Interfaces**: Command-line arguments or interactive mode
- 💾 **Result Export**: Save results to text files

## Prerequisites

1. **Environment Setup**: Ensure your `.env` file is configured with:
   ```
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=ai-hiring-portal
   OPENAI_MODEL=text-embedding-3-large
   ```

2. **Resume Embeddings**: Run the resume embedding script first:
   ```bash
   python src/scripts/embed_resumes_to_pinecone.py
   ```

## Usage

### Option 1: Simple Interactive Mode
```bash
python src/scripts/simple_job_matcher.py
```

This will prompt you for:
- Job Title
- Required Skills/Keywords
- Additional Requirements (optional)
- Job Description (optional)
- Number of matches (default: 3)

### Option 2: Command Line Arguments
```bash
python src/scripts/job_matching.py \
    --job-title "Senior Python Developer" \
    --keywords "Python, Django, REST API, PostgreSQL, AWS" \
    --requirements "5+ years experience, Bachelor's degree" \
    --description "We are looking for an experienced Python developer..." \
    --top-k 3
```

### Option 3: Interactive Mode (Advanced)
```bash
python src/scripts/job_matching.py
```
(Run without arguments for interactive prompts)

## Example Usage

### Example 1: Software Developer Position
```bash
python src/scripts/simple_job_matcher.py
```
Input:
- Job Title: `Software Engineer`
- Keywords: `Python, Machine Learning, TensorFlow, Git`
- Requirements: `3+ years experience`
- Description: `Building AI applications`

### Example 2: Data Scientist Position
```bash
python src/scripts/job_matching.py \
    -j "Data Scientist" \
    -k "Python, ML, Statistics, Pandas, Scikit-learn" \
    -r "PhD in Data Science or related field" \
    -t 5
```

## Output Format

The system returns results in this format:

```
🎯 TOP MATCHING CANDIDATES FOR: SENIOR PYTHON DEVELOPER
================================================================================

🏆 RANK #1
📂 Category: INFORMATION-TECHNOLOGY
🆔 Resume ID: 42
📊 Cosine Similarity: 0.8534
📈 Match Percentage: 92.67%
🎯 Status: ✅ EXCELLENT MATCH
📝 Text Preview: Senior Software Engineer with 8 years experience in Python...
------------------------------------------------------------

🏆 RANK #2
📂 Category: ENGINEERING
🆔 Resume ID: 156
📊 Cosine Similarity: 0.8102
📈 Match Percentage: 90.51%
🎯 Status: ✅ EXCELLENT MATCH
📝 Text Preview: Full-stack developer specializing in Django and React...
------------------------------------------------------------
```

## Matching Criteria

### Match Status Levels:
- **✅ EXCELLENT MATCH**: 80%+ similarity
- **🟢 GOOD MATCH**: 70-79% similarity
- **🟡 MODERATE MATCH**: 60-69% similarity
- **🔴 WEAK MATCH**: <60% similarity

### Similarity Calculation:
- Uses cosine similarity between job description and resume embeddings
- Converted to percentage: `((cosine_score + 1) / 2) * 100`
- Ranges from 0% (completely opposite) to 100% (perfect match)

## File Structure

```
src/scripts/
├── job_matching.py          # Main job matching engine
├── simple_job_matcher.py    # Simple interactive interface
├── embed_resumes_to_pinecone.py  # Resume embedding pipeline
└── README_job_matching.md   # This documentation
```

## Configuration Options

### Environment Variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Pinecone index name (default: ai-hiring-portal)
- `OPENAI_MODEL`: Embedding model (default: text-embedding-3-large)
- `PINECONE_ENVIRONMENT`: Pinecone environment (default: us-east-1)

### Script Parameters:
- `--job-title, -j`: Job title (required)
- `--keywords, -k`: Skills and keywords (required)
- `--requirements, -r`: Job requirements (optional)
- `--description, -d`: Job description (optional)
- `--top-k, -t`: Number of matches to return (default: 3)

## Troubleshooting

### Common Issues:

1. **No matches found**:
   - Check if resume embeddings exist in Pinecone
   - Verify index name matches environment variable
   - Try broader keywords

2. **API errors**:
   - Verify API keys in `.env` file
   - Check OpenAI API quota and billing
   - Ensure Pinecone index exists

3. **Low match scores**:
   - Try more specific keywords
   - Include relevant technical terms
   - Check if target skills exist in resume database

### Debug Mode:
Add print statements or use Python debugger to inspect:
- Embedding vectors
- Query response structure
- Metadata content

## Performance Notes

- **Embedding Creation**: ~1-2 seconds per job description
- **Search Time**: ~0.5 seconds for similarity search
- **Memory Usage**: Minimal (vectors stored in Pinecone)
- **Rate Limits**: Respects OpenAI API rate limits

## Next Steps

Consider enhancing with:
1. **Weighted Keywords**: Different importance for different skills
2. **Experience Level Matching**: Junior/Senior level filtering
3. **Location-based Filtering**: Geographic preferences
4. **Salary Range Matching**: Compensation alignment
5. **Industry-specific Models**: Specialized embeddings per domain
