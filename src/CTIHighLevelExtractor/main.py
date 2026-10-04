#!/usr/bin/env python3
"""
CTIHighLevelExtractor main program entry point
"""

import argparse
import json
import sys
from pathlib import Path

from .core.sentence_splitter import SentenceSplitter
from .core.llm_provider import create_llm_provider
from .core.entity_extractor import EntityExtractor, EntityMerger, EntityPrecisionVerifier


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="CTIHighLevelExtractor - extract contextual entities from CTI reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m CTIHighLevelExtractor --input_txt report.txt --output_dir ./output/ --model gpt-4o
  python -m CTIHighLevelExtractor --input_txt report.txt --output_dir ./output/ --model gpt-4o --chunk_size 40
        """
    )
    
    parser.add_argument(
        "--input_txt",
        type=str,
        required=True,
        help="Path to the CTI report plain-text (.txt) file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory path"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="LLM model name (default: gpt-5; gpt-5-mini is also supported)"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai"],
        help="LLM provider type (default: openai)"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=40,
        help="Number of sentences per chunk (default: 40)"
    )
    
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=2,
        help="Number of overlapping sentences between chunks (default: 2)"
    )
    
    parser.add_argument(
        "--token_overlap_threshold",
        type=float,
        default=0.6,
        help="Token overlap threshold (default: 0.6)"
    )
    
    args = parser.parse_args()
    
    # Validate the input file
    input_path = Path(args.input_txt)
    if not input_path.exists():
        print(f"❌ Input file does not exist: {args.input_txt}")
        sys.exit(1)
    
    if not input_path.suffix.lower() == '.txt':
        print(f"⚠️  Warning: input file is not a .txt file: {args.input_txt}")
    
    # Create the output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename: same name as the input txt, with a .json extension
    output_filename = input_path.stem + ".json"
    output_path = output_dir / output_filename
    
    # Read the input file
    print("📄 Reading input file...")
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Sentence splitting (consistent with CTITagger)
    print("✂️  Splitting sentences...")
    splitter = SentenceSplitter()
    sentences = splitter.split_sentences(raw_text)
    print(f"✅ Split into {len(sentences)} sentences")
    
    # Create the LLM provider
    print(f"🤖 Initializing LLM provider (model: {args.model})...")
    try:
        llm_provider = create_llm_provider(
            provider_type=args.provider,
            model=args.model
        )
    except Exception as e:
        print(f"❌ Failed to initialize LLM provider: {e}")
        sys.exit(1)
    
    # Create the entity extractor
    extractor = EntityExtractor(
        llm_provider=llm_provider,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        token_overlap_threshold=args.token_overlap_threshold
    )
    
    # Extract entities
    print("\n🔍 Extracting contextual entities...")
    print(f"   Config: chunk_size={args.chunk_size}, overlap={args.overlap_size}, threshold={args.token_overlap_threshold}")
    print("-" * 60)
    
    entities = extractor.extract(sentences)
    print(f"\n✅ Extraction done: {len(entities)} entities (before deduplication)")
    
    # Deduplicate and merge
    print("\n🔄 Deduplicating and merging entities...")
    merger = EntityMerger()
    merged_entities = merger.merge_entities(entities, sentences)
    print(f"✅ Merge done: {len(merged_entities)} unique entities")

    # Precision check after merge (improves precision)
    print("\n🧪 Running precision check on entities...")
    verifier = EntityPrecisionVerifier(
        llm_provider=llm_provider,
        max_span_sentences=3
    )
    final_entities = verifier.verify_entities(merged_entities)
    filtered_count = len(merged_entities) - len(final_entities)
    print(f"✅ Precision check done: kept {len(final_entities)} entities (filtered {filtered_count})")
    
    # Build the output JSON
    output_data = {
        "input_txt": str(input_path),
        "model": args.model,
        "entities": final_entities
    }
    
    # Save the output
    print(f"\n💾 Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Print the summary
    print("\n" + "=" * 60)
    print("📊 Processing Summary")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Model: {args.model}")
    print(f"Sentences: {len(sentences)}")
    print(f"Extracted entities (before deduplication): {len(entities)}")
    print(f"Unique entities (after merge): {len(merged_entities)}")
    print(f"Final entities (after precision check): {len(final_entities)}")
    
    # Count by type
    type_counts = {}
    for entity in final_entities:
        entity_type = entity["entity_type"]
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    
    print("\nCounts by type:")
    for entity_type, count in sorted(type_counts.items()):
        print(f"  {entity_type}: {count}")
    
    print("\n✅ Processing complete!")


if __name__ == "__main__":
    main()
