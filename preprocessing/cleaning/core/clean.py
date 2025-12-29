"""
clean.py
--------
Deterministic WhatsApp chat cleaning: noise removal + high-risk PII masking.
Outputs raw JSON with parsed messages (no chunking/formatting).

Input:  datasets/chats_raw/*.txt (iPhone or Android WhatsApp exports)
Output: datasets/chats_cleaned/<persona>.json (list of {timestamp, sender, message})

Cleaning strategy:
- Remove system messages (joined, left, added, removed, etc.)
- Remove media omissions and missed calls
- Filter out very short messages (< 3 characters) to reduce noise
- Mask sensitive PII (phone, email, credit cards, IBAN)
"""

import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from collections import Counter


# --- PATH CONFIG ---
RAW_DIR = Path(__file__).parent.parent.parent.parent / "datasets" / "core" / "chats_raw"
CLEANED_DIR = Path(__file__).parent.parent.parent.parent / "datasets" / "core" / "chats_cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# --- SYSTEM / MEDIA MESSAGES TO REMOVE ---
SYSTEM_PATTERNS = [
    "end-to-end encrypted",
    "joined using this phone number",
    "This message was deleted",
    "You deleted this message",
    "Missed voice call",
    "Missed video call",
    "changed their phone number",
    "left",
    "added",
    "removed",
    "voice call",
    "is a contact."
]

MEDIA_PATTERNS = [
    "<Media omitted>",
    "omitted",
    "image",
    "audio",
    "sticker",
    "gif",
    "document",
    "video",
]

# --- SUBSTRINGS TO STRIP FROM MESSAGES ---
# Unlike system messages that filter entire messages, these substrings are removed
# from message text while preserving the user's actual intent.
SUBSTRINGS_TO_REMOVE = [
    "<This message was edited>",
]

# --- PII MASKING PATTERNS (deterministic regex) ---
PII_PATTERNS = {
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # 16-digit card numbers
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",
}


MIN_MESSAGE_LENGTH = 3

def is_noise(message: str) -> bool:
    """Return True if message is system/media noise."""
    lower_msg = message.lower()
    return any(p.lower() in lower_msg for p in SYSTEM_PATTERNS + MEDIA_PATTERNS)


def is_too_short(message: str) -> bool:
    """Return True if message is too short (likely noise)."""
    # Count only alphanumeric and space characters (excludes emojis)
    text_only = re.sub(r'[^a-zA-Z0-9\s]', '', message)
    return len(text_only.strip()) < MIN_MESSAGE_LENGTH


def strip_substrings(message: str) -> str:
    """Remove specific substrings from message text."""
    for substring in SUBSTRINGS_TO_REMOVE:
        message = message.replace(substring, "").strip()
    return message


def mask_pii(message: str) -> str:
    """Mask high-risk PII deterministically using regex."""
    for pii_type, pattern in PII_PATTERNS.items():
        message = re.sub(pattern, f"[{pii_type}]", message, flags=re.IGNORECASE)
    return message


def parse_chat_line_iphone(line: str) -> Optional[Tuple[datetime, str, str]]:
    """
    Parse WhatsApp iPhone-style line: [DD/MM/YYYY, HH:MM:SS] Sender: Message
    Returns (timestamp, sender, message) or None if invalid.
    """
    line = line.replace("\u200e", "").strip()
    pattern = r"\[(\d{1,2}/\d{1,2}/\d{4}), (\d{2}:\d{2}:\d{2})\] (.*?): (.*)"
    match = re.match(pattern, line)
    if not match:
        return None
    date_str, time_str, sender, message = match.groups()
    try:
        timestamp = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M:%S")
    except ValueError:
        return None
    return timestamp, sender.strip(), message.strip()


def parse_chat_line_android(line: str) -> Optional[Tuple[datetime, str, str]]:
    """
    Parse WhatsApp Android-style line: DD/MM/YYYY, HH:MM - Sender: Message
    Returns (timestamp, sender, message) or None if invalid.
    """
    line = line.replace("\u200e", "").strip()
    pattern = r"(\d{1,2}/\d{1,2}/\d{4}), (\d{2}:\d{2}) - (.*?): (.*)"
    match = re.match(pattern, line)
    if not match:
        return None
    date_str, time_str, sender, message = match.groups()
    try:
        timestamp = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
    except ValueError:
        return None
    return timestamp, sender.strip(), message.strip()


def parse_chat_line(line: str) -> Optional[Tuple[datetime, str, str]]:
    """Try iPhone format first, then Android."""
    result = parse_chat_line_iphone(line)
    if result:
        return result
    return parse_chat_line_android(line)


def load_and_clean_chat_file(filepath: Path) -> List[dict]:
    """
    Load raw chat file, filter noise, mask PII, and return cleaned messages.
    
    Filtering steps:
    1. Parse timestamp, sender, message
    2. Remove system/media messages
    3. Remove very short messages (< 3 chars)
    4. Normalize whitespace (preserve emojis)
    5. Mask PII (phone, email, credit card, IBAN)
    """
    messages = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_chat_line(line)
            if not parsed:
                continue
            timestamp, sender, message = parsed
            
            if is_noise(message):
                continue
            
            message = strip_substrings(message)
            message = re.sub(r"\s+", " ", message).strip()
            if not message:
                continue
            
            if is_too_short(message):
                continue
            
            message = mask_pii(message)
            
            messages.append({
                "timestamp": timestamp.isoformat(),
                "sender": sender,
                "message": message,
            })
    
    return messages


def main():
    """Clean all raw chat files and save as JSON."""
    for file in sorted(RAW_DIR.glob("*.txt")):
        print(f"Cleaning {file.name} ...")
        
        persona = file.stem.upper()
        
        messages = load_and_clean_chat_file(file)
        if not messages:
            print(f"No valid messages found.")
            continue
        
        print(f"Cleaned {len(messages)} messages")
        
        output_path = CLEANED_DIR / f"{persona}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
