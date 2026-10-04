#!/usr/bin/env python3
"""
CTITagger main program entry point
"""

import argparse
import sys
from pathlib import Path
from .core.ctitagger import CTITagger
from .core.span_locator import SpanLocatorConfig


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="CTITagger - CTI report processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py --report report.txt --mitre mitre.json --output ./output/
  python main.py --report report.txt --mitre mitre.json --model gpt-5 --threshold 0.7
        """
    )
    
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to the CTI report file (.txt file)"
    )
    
    parser.add_argument(
        "--mitre",
        type=str,
        required=True,
        help="Path to the MITRE ATT&CK JSON file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory path (optional; nothing is saved by default. The output file name matches the input txt file name with a .json extension)"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai"],
        help="LLM provider type (default: openai)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model name (default: gpt-4o; gpt-5, gpt-5-mini, etc. are supported)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.71,
        help="TTP verification threshold (default: 0.71)"
    )
    
    parser.add_argument(
        "--lmax",
        type=int,
        default=5,
        help="Maximum Span Locator window length (default: 5)"
    )
    
    parser.add_argument(
        "--high-freq-threshold",
        type=float,
        default=0.01,
        help="High-frequency token threshold (default: 0.01, i.e. the top 1%%)"
    )
    
    parser.add_argument(
        "--enable-ngram",
        action="store_true",
        default=True,
        help="Enable the n-gram match bonus (default: True)"
    )
    
    parser.add_argument(
        "--disable-ngram",
        action="store_false",
        dest="enable_ngram",
        help="Disable the n-gram match bonus"
    )
    
    args = parser.parse_args()
    
    # Check that the files exist
    report_path = Path(args.report)
    mitre_path = Path(args.mitre)
    
    if not report_path.exists():
        print(f"❌ Report file does not exist: {args.report}")
        sys.exit(1)
    
    if not mitre_path.exists():
        print(f"❌ MITRE JSON file does not exist: {args.mitre}")
        sys.exit(1)
    
    # Build the configuration
    span_config = SpanLocatorConfig(
        lmax=args.lmax,
        high_freq_threshold=args.high_freq_threshold,
        enable_ngram=args.enable_ngram
    )
    
    # Create the CTITagger instance
    try:
        tagger = CTITagger(
            provider_type=args.provider,
            model=args.model,
            span_config=span_config,
            ttp_threshold=args.threshold
        )
    except Exception as e:
        print(f"❌ Failed to initialize CTITagger: {e}")
        sys.exit(1)
    
    # Resolve the output path
    output_path = None
    if args.output:
        output_arg = Path(args.output)
        # If the output path has an extension (e.g. .json), treat it as a file path
        # Otherwise treat it as a directory and generate the file name automatically
        if output_arg.suffix:
            # A full file path was given; use it directly
            output_path = output_arg
        else:
            # Treat it as a directory; use the input file's base name with a .json extension
            output_filename = report_path.stem + ".json"
            output_path = output_arg / output_filename
    
    # Process the report
    print("🚀 Starting CTI report processing...")
    print(f"📄 Report file: {args.report}")
    print(f"📚 MITRE file: {args.mitre}")
    print(f"🤖 Model: {args.model}")
    print(f"⚙️  Threshold: {args.threshold}")
    if output_path:
        print(f"💾 Output file: {output_path}")
    print("-" * 60)
    
    try:
        results = tagger.process_file(
            report_file_path=str(report_path),
            mitre_json_path=str(mitre_path),
            output_path=str(output_path) if output_path else None
        )
        
        # Print the summary
        print("\n" + "=" * 60)
        print("📊 Processing summary")
        print("=" * 60)
        print(f"Total behaviors: {len(results)}")
        
        total_final_ttps = sum(len(r["final_ttps"]) for r in results)
        print(f"Total final TTPs: {total_final_ttps}")
        
        for i, result in enumerate(results, 1):
            print(f"\nBehavior {i}:")
            print(f"  Text: {result['behavior_text'][:80]}...")
            print(f"  Span range: {result['best_span_range']}")
            print(f"  Candidate TTPs: {len(result['ttp_candidates'])}")
            print(f"  Final TTPs: {len(result['final_ttps'])}")
            if result['final_ttps']:
                print(f"  Final TTP IDs: {[t['ttp_id'] for t in result['final_ttps']]}")
        
        print("\n✅ Processing complete!")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
