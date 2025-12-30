"""
Core Dataset Processing
Converts cleaned JSON chat files into fine-tuning ready datasets.
Outputs ChatML format with persona tags and LLM-based semantic chunking.

Input:  datasets/core/chats_cleaned/<persona>.json
Output: datasets/core/chats_processed/conversations.jsonl (ChatML format)
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import requests

# --- LOGGING CONFIG ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing/process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- PATH CONFIG ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
CLEANED_DIR = PROJECT_ROOT / "datasets" / "core" / "chats_cleaned"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "core" / "chats_processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- CHUNKING CONFIG ---
AUTO_GROUP_MINUTES = 15
LLM_QUERY_MIN_MINUTES = 15
LLM_QUERY_MAX_MINUTES = 240
SLEEP_START_HOUR = 0
SLEEP_END_HOUR = 10

# --- LLM CHUNKING CONFIG ---
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "https://enhanced-accepted-jaybird.ngrok-free.app/api/chat")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3:8b")

# --- TEST MODE CONFIG ---
TEST_MODE = False
TEST_MESSAGE_LIMIT = 200

# --- PERSONA CONFIG ---
YOUR_NAME = "Festus"


def is_in_sleep_hours(dt: datetime) -> bool:
    """Check if datetime is during sleep hours (12am-10am)."""
    return SLEEP_START_HOUR <= dt.hour < SLEEP_END_HOUR


def ask_llm_about_continuation(conversation_so_far: List[dict], new_message: dict) -> bool:
    """
    Ask LLM if new_message continues the conversation.
    
    Returns True if message belongs to same conversation, False if new conversation.
    """
    try:
        context = "Previous conversation:\n"
        for msg in conversation_so_far[-10:]:
            sender = "User" if msg["sender"] == conversation_so_far[0]["sender"] else "Assistant"
            context += f"[{msg['timestamp']}] {sender}: {msg['message']}\n"
        
        context += f"\nNew message at [{new_message['timestamp']}]:\n"
        sender = "User" if new_message["sender"] == conversation_so_far[0]["sender"] else "Assistant"
        context += f"{sender}: {new_message['message']}\n"
        
        prompt = f"""{context}
Does this new message continue the same conversation, or does it start a new topic/conversation?
Answer with only "YES" or "NO"."""
        
        logger.debug(f"Sending prompt to LLM: {prompt[:100]}...")
        logger.debug(f"LLM Endpoint: {LLM_ENDPOINT}")
        logger.debug(f"LLM Model: {LLM_MODEL}")
        
        response = requests.post(
            LLM_ENDPOINT,
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=30
        )
        
        logger.debug(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("message", {}).get("content", "").strip().upper()
            logger.info(f"LLM answer: {answer}")
            return "YES" in answer
        else:
            logger.error(f"LLM error: {response.status_code}")
            logger.error(f"Response text: {response.text}")
            print(f"LLM error: {response.status_code}, falling back to time-based decision")
            return False
    except Exception as e:
        logger.exception(f"LLM request failed: {e}")
        print(f"LLM request failed: {e}, falling back to time-based decision")
        return False


def chunk_by_llm(messages: List[dict]) -> List[List[dict]]:
    """
    Split messages into chunks using LLM with smart time-based heuristics.
    
    Algorithm:
    1. If time gap < 15 min → auto-group (same conversation)
    2. If time gap 15 min - 4 hours → ask LLM
    3. If time gap > 4 hours:
       - During sleep (12am-10am) → ask LLM
       - During waking hours → force split
    """
    if not messages:
        return []
    
    if len(messages) < 2:
        return [messages]
    
    chunks = []
    current_chunk = [messages[0]]
    
    for i in range(1, len(messages)):
        prev_msg = messages[i - 1]
        curr_msg = messages[i]
        
        prev_time = datetime.fromisoformat(prev_msg["timestamp"])
        curr_time = datetime.fromisoformat(curr_msg["timestamp"])
        time_diff = curr_time - prev_time
        time_diff_minutes = time_diff.total_seconds() / 60
        
        should_continue = False
        
        if time_diff_minutes < AUTO_GROUP_MINUTES:
            should_continue = True
        elif time_diff_minutes <= LLM_QUERY_MAX_MINUTES:
            should_continue = ask_llm_about_continuation(current_chunk, curr_msg)
        else:
            if is_in_sleep_hours(curr_time):
                should_continue = ask_llm_about_continuation(current_chunk, curr_msg)
            else:
                should_continue = False
        
        if should_continue:
            current_chunk.append(curr_msg)
        else:
            chunks.append(current_chunk)
            current_chunk = [curr_msg]
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def chunk_to_chatml(chunk: List[dict], persona: str) -> Optional[dict]:
    """
    Convert a chunk (list of messages) to ChatML format.
    
    Role assignment:
    - YOUR_NAME is always "assistant"
    - Other person is always "user"
    
    Processing:
    1. Trim leading assistant messages until we find a user message
    2. Merge consecutive messages from the same sender
    3. Skip chunks with no assistant messages
    
    Returns None if chunk doesn't meet criteria.
    """
    if len(chunk) < 2:
        return None
    
    # Trim leading assistant messages until we find a user message
    start_idx = 0
    for i, msg in enumerate(chunk):
        if msg["sender"] != YOUR_NAME:
            start_idx = i
            break
    else:
        logger.debug(f"Skipping chunk: no user messages found.")
        return None
    
    chunk = chunk[start_idx:]
    
    # Merge consecutive messages from same sender
    merged_messages = []
    current_role = None
    current_content = []
    
    for msg in chunk:
        role = "assistant" if msg["sender"] == YOUR_NAME else "user"
        
        if role == current_role:
            current_content.append(msg["message"])
        else:
            if current_role is not None:
                merged_messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content),
                })
            current_role = role
            current_content = [msg["message"]]
    
    # Append final message
    if current_role is not None:
        merged_messages.append({
            "role": current_role,
            "content": "\n".join(current_content),
        })
    
    # Skip chunks with no assistant messages
    has_assistant = any(msg["role"] == "assistant" for msg in merged_messages)
    if not has_assistant:
        logger.debug(f"Skipping chunk: no assistant messages.")
        return None
    
    return {
        "messages": merged_messages,
        "persona": persona,
        "timestamp_start": chunk[0]["timestamp"],
        "timestamp_end": chunk[-1]["timestamp"],
    }


def process_cleaned_file(filepath: Path) -> List[dict]:
    """
    Load cleaned JSON, chunk by LLM, and convert to ChatML format.
    Persona is inferred from filename (e.g., sister.json -> "sister").
    Saves conversations to persona-specific JSONL file.
    """
    print(f"Processing {filepath.name} ...")
    
    persona = filepath.stem.lower()
    
    # Load cleaned messages
    with open(filepath, "r", encoding="utf-8") as f:
        messages = json.load(f)
    
    if not messages:
        print(f"No messages found.")
        return []
    
    print(f"Loaded {len(messages)} total messages")
    
    # TEST MODE: Limit messages for quick testing
    if TEST_MODE:
        messages = messages[:TEST_MESSAGE_LIMIT]
        print(f"\n{'='*60}")
        print(f"TEST MODE: Processing only first {len(messages)} messages")
        print(f"{'='*60}")
        print("\nMessage List:")
        for idx, msg in enumerate(messages, 1):
            print(f"\n{idx}. [{msg['timestamp']}] {msg['sender']}")
            print(f"   Message: {msg['message'][:80]}{'...' if len(msg['message']) > 80 else ''}")
        print(f"\n{'='*60}\n")
    
    # Chunk by LLM with smart time-based heuristics
    messages.sort(key=lambda m: m["timestamp"])
    chunks = chunk_by_llm(messages)
    print(f"Split into {len(chunks)} conversation chunks (LLM + time-based)")
    
    # Convert to ChatML
    chatml_conversations = []
    for chunk in chunks:
        conv = chunk_to_chatml(chunk, persona)
        if conv:
            chatml_conversations.append(conv)
    
    print(f"Converted to {len(chatml_conversations)} ChatML conversations")
    
    # Save to persona-specific JSONL file
    persona_output_file = PROCESSED_DIR / f"{persona}.jsonl"
    with open(persona_output_file, "w", encoding="utf-8") as f:
        for conv in chatml_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    print(f"Saved to {persona_output_file}")
    
    return chatml_conversations


def main():
    """Process cleaned chat files for specified persona and save as JSONL."""
    global TEST_MODE
    
    # Get persona from command line argument
    persona = None
    if len(sys.argv) > 1:
        if not sys.argv[1].startswith("-"):
            persona = sys.argv[1]
    
    # Check for test mode flag
    if "--test" in sys.argv or "-t" in sys.argv:
        TEST_MODE = True
        print("🧪 TEST MODE ENABLED - Processing only first 200 messages\n")
    
    all_conversations = []
    
    if persona:
        # Process single persona (used for test mode)
        persona_file = CLEANED_DIR / f"{persona.upper()}.json"
        if not persona_file.exists():
            print(f"Error: File not found: {persona_file}")
            print(f"Available files:")
            for f in sorted(CLEANED_DIR.glob("*.json")):
                print(f"  - {f.stem}")
            return
        
        conversations = process_cleaned_file(persona_file)
        all_conversations.extend(conversations)
    else:
        # Process all personas
        persona_files = sorted(CLEANED_DIR.glob("*.json"))
        if not persona_files:
            print(f"Error: No cleaned chat files found in {CLEANED_DIR}")
            return
        
        print(f"Processing {len(persona_files)} personas...\n")
        for persona_file in persona_files:
            print(f"Processing {persona_file.stem}...")
            conversations = process_cleaned_file(persona_file)
            all_conversations.extend(conversations)
    
    # Print summary stats
    personas = {}
    for conv in all_conversations:
        persona = conv["persona"]
        personas[persona] = personas.get(persona, 0) + 1
    
    print(f"\nSummary by persona:")
    for persona, count in sorted(personas.items()):
        print(f"  {persona}: {count} conversations")
    
    print(f"\nPersona-specific files saved to {PROCESSED_DIR}/")
    print(f"Use 'make combine-core' to merge all personas into a single training file.")


if __name__ == "__main__":
    main()
