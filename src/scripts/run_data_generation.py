#!/usr/bin/env python3
"""
Simple runner script for synthetic data generation.
This script provides an easy way to run the data generation with different parameters.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scripts.data_generation import SyntheticDataGenerator

def main():
    """Run synthetic data generation with user-friendly interface."""
    
    print("🚀 AI Hiring Portal - Synthetic Data Generator")
    print("=" * 50)
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr set it in your shell profile (~/.zshrc, ~/.bash_profile, etc.)")
        return
    
    print("✅ OpenAI API key found")
    
    # Get user preferences
    try:
        num_records = input("\n📊 How many records to generate? (default: 1000): ").strip()
        num_records = int(num_records) if num_records else 1000
        
        batch_size = input("📦 Batch size (records per API call, default: 25): ").strip()
        batch_size = int(batch_size) if batch_size else 25
        
        custom_output = input("💾 Custom output file path (press Enter for default): ").strip()
        output_file = custom_output if custom_output else None
        
    except KeyboardInterrupt:
        print("\n\n👋 Generation cancelled by user.")
        return
    except ValueError:
        print("❌ Invalid input. Using default values.")
        num_records = 1000
        batch_size = 25
        output_file = None
    
    print(f"\n🔧 Configuration:")
    print(f"   Records: {num_records}")
    print(f"   Batch size: {batch_size}")
    print(f"   Output: {'Default location' if not output_file else output_file}")
    
    confirm = input("\n▶️  Start generation? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("👋 Generation cancelled.")
        return
    
    # Initialize and run generator
    try:
        generator = SyntheticDataGenerator(api_key=api_key, batch_size=batch_size)
        result_file = generator.generate_dataset(
            total_records=num_records,
            output_file=output_file
        )
        
        print("\n" + "=" * 50)
        print("🎉 SUCCESS! Data generation completed!")
        print(f"📁 File saved to: {result_file}")
        print("\n💡 Next steps:")
        print("   1. Review the generated data for quality")
        print("   2. Use the data for training your ML models")
        print("   3. Adjust parameters and re-run if needed")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Generation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during generation: {str(e)}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    main()
