"""
Combine persona-specific JSONL files into a single training file.

Input:  datasets/core/chats_processed/<persona>.jsonl
Output: datasets/core/chats_processed/conversations.jsonl
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "core" / "chats_processed"


def combine_personas():
    """Combine all persona-specific JSONL files into a single training file."""
    persona_files = sorted(PROCESSED_DIR.glob("*.jsonl"))
    
    if not persona_files:
        print(f"Error: No persona JSONL files found in {PROCESSED_DIR}")
        return
    
    print(f"Found {len(persona_files)} persona files:")
    for f in persona_files:
        print(f"  - {f.name}")
    
    all_conversations = []
    
    for persona_file in persona_files:
        print(f"\nReading {persona_file.name}...")
        with open(persona_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    conv = json.loads(line)
                    all_conversations.append(conv)
        print(f"  Loaded {len(all_conversations)} conversations so far")
    
    if not all_conversations:
        print("No conversations found.")
        return
    
    # Sort by persona, then by timestamp
    all_conversations.sort(key=lambda c: (c["persona"], c["timestamp_start"]))
    
    # Save combined file
    output_file = PROCESSED_DIR / "conversations.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Combined {len(all_conversations)} conversations into {output_file}")
    
    # Print summary stats
    personas = {}
    for conv in all_conversations:
        persona = conv["persona"]
        personas[persona] = personas.get(persona, 0) + 1
    
    print(f"\nSummary by persona:")
    for persona, count in sorted(personas.items()):
        print(f"  {persona}: {count} conversations")


if __name__ == "__main__":
    combine_personas()
