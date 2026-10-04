#!/usr/bin/env python3
"""
CTI IOC Extractor command-line entry point
"""

import os
import sys
import json
from pathlib import Path

# Import the LLM Provider
try:
    from CTITagger.core.llm_provider import create_llm_provider
except ImportError:
    create_llm_provider = None

from .extractor import CTIIOCExtractor


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CTI IOC Extractor - extract IoCs from CTI reports")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input txt file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--model", "-m", type=str, default="gpt-5-mini", help="LLM model name (default: gpt-5-mini; options: gpt-5, gpt-5-mini, etc.)")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (if not provided, read from the environment variable)")
    parser.add_argument(
        "--pre-verification-output-dir",
        type=str,
        default=None,
        help=(
            "Optional: record the IoC candidates before second-pass verification as a separate JSON file; "
            "does not change the normal output or the verification flow"
        ),
    )
    
    args = parser.parse_args()
    
    # Check the input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}")
        sys.exit(1)
    
    # Read the input file
    print(f"📖 Reading input file: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Create the output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build the output filename (same as the input filename, but with a .json extension)
    output_filename = input_path.stem + ".json"
    output_path = output_dir / output_filename
    
    # Initialize the LLM Provider (if available)
    llm_provider = None
    if create_llm_provider:
        try:
            api_key = args.api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                llm_provider = create_llm_provider(
                    provider_type="openai",
                    model=args.model,
                    api_key=api_key
                )
                print(f"✅ LLM Provider initialized successfully: {args.model}")
            else:
                print("⚠️  No OpenAI API key provided; only regex extraction will be used")
        except Exception as e:
            print(f"⚠️  LLM Provider initialization failed: {e}; only regex extraction will be used")
    
    # Create the extractor
    extractor = CTIIOCExtractor(llm_provider=llm_provider, model=args.model)
    
    # Extract IoCs
    print("\n" + "="*50)
    print("Starting IoC extraction...")
    print("="*50 + "\n")
    
    iocs = extractor.extract(
        text,
        record_pre_verification=bool(args.pre_verification_output_dir),
    )
    
    # Save the results
    result = {
        "metadata": {
            "input_file": str(input_path),
            "total_sentences": len(extractor.split_sentences(text)),
            "total_iocs": len(iocs),
            "extraction_methods": ["regex", "llm"] if llm_provider else ["regex"]
        },
        "iocs": iocs
    }
    
    print(f"\n💾 Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if args.pre_verification_output_dir:
        pre_verification_dir = Path(args.pre_verification_output_dir)
        pre_verification_dir.mkdir(parents=True, exist_ok=True)
        pre_verification_path = pre_verification_dir / output_filename
        pre_verification_result = {
            "metadata": {
                "input_file": str(input_path),
                "report_id": input_path.stem,
                "model": args.model,
                "stage": "pre_second_pass_verification",
                "total_sentences": len(extractor.split_sentences(text)),
                "candidate_occurrence_count": len(extractor.pre_verification_iocs),
                "merged_candidate_count": len(
                    extractor.pre_verification_merged_iocs
                ),
                "recording_only": True,
            },
            "candidate_occurrences": extractor.pre_verification_iocs,
            "merged_candidates": extractor.pre_verification_merged_iocs,
        }
        with open(pre_verification_path, "w", encoding="utf-8") as f:
            json.dump(pre_verification_result, f, indent=2, ensure_ascii=False)
        print(f"📝 Pre-verification candidates recorded to: {pre_verification_path}")
    
    print(f"✅ Done! Extracted {len(iocs)} IoCs in total")


if __name__ == "__main__":
    main()
