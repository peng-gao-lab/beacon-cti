#!/usr/bin/env python3
"""
CTIAugmentor CLI entry point
"""

import argparse
import json
from pathlib import Path
from .pipeline import CTIAugmentorPipeline


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="CTIAugmentor: Build High Level ↔ TTP and IoC ↔ TTP edges"
    )
    
    parser.add_argument(
        "--highlevel_json",
        type=str,
        required=True,
        help="Path to CTIHighLevelExtractor output JSON"
    )
    
    parser.add_argument(
        "--ioc_json",
        type=str,
        required=True,
        help="Path to CTIIoCExtractor output JSON"
    )
    
    parser.add_argument(
        "--ttp_json",
        type=str,
        required=True,
        help="Path to CTITagger output JSON"
    )
    
    parser.add_argument(
        "--input_txt",
        type=str,
        required=True,
        help="Path to original CTI report text file"
    )

    parser.add_argument(
        "--mitre_json",
        type=str,
        required=True,
        help="Path to MITRE ATT&CK JSON file (used to inject TTP descriptions into verifier prompt)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--delta",
        type=int,
        default=0,
        help="Distance threshold for candidate generation (default: 0)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="LLM model name (default: gpt-5, supports gpt-4o, gpt-5-mini, etc.)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for edge verification (default: 0.8)"
    )
    
    parser.add_argument(
        "--check_split",
        action="store_true",
        help="Enable sentence alignment check (default: False)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in [args.highlevel_json, args.ioc_json, args.ttp_json, args.input_txt, args.mitre_json]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename (same as input_txt, with .json extension)
    input_txt_path = Path(args.input_txt)
    output_filename = input_txt_path.stem + ".json"
    output_path = output_dir / output_filename
    
    print("=" * 60)
    print("CTIAugmentor Pipeline")
    print("=" * 60)
    print(f"High Level JSON: {args.highlevel_json}")
    print(f"IoC JSON: {args.ioc_json}")
    print(f"TTP JSON: {args.ttp_json}")
    print(f"Input TXT: {args.input_txt}")
    print(f"MITRE JSON: {args.mitre_json}")
    print(f"Output: {output_path}")
    print(f"Delta: {args.delta}")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Check Split: {args.check_split}")
    print("=" * 60)
    print()
    
    # Initialize pipeline
    pipeline = CTIAugmentorPipeline(
        delta=args.delta,
        verifier_threshold=args.threshold,
        model=args.model,
        check_split=args.check_split
    )
    
    # Process
    try:
        result = pipeline.process(
            highlevel_json_path=args.highlevel_json,
            ioc_json_path=args.ioc_json,
            ttp_json_path=args.ttp_json,
            input_txt_path=args.input_txt,
            mitre_json_path=args.mitre_json
        )
        
        # Save output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print()
        print("=" * 60)
        print(f"✅ Success! Results saved to: {output_path}")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Error: {str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
