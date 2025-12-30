"""
Core Labelling Application
Human labelling UI for cleaned WhatsApp conversations.

Supports multiple labelling modes (extensible for future VLM support).
"""

import json
import os
from functools import wraps
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import logging

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets" / "core"
CLEANED_DIR = DATASETS_DIR / "chats_cleaned"
LABELED_DIR = DATASETS_DIR / "chats_labeled"
LABELED_DIR.mkdir(parents=True, exist_ok=True)

VLM_DATASETS_DIR = PROJECT_ROOT / "datasets" / "vlm"
VLM_CLEANED_DIR = VLM_DATASETS_DIR / "cleaned"
VLM_LABELED_DIR = VLM_DATASETS_DIR / "labeled"
VLM_LABELED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = LABELED_DIR / "conversations.jsonl"
VLM_OUTPUT_FILE = VLM_LABELED_DIR / "labeled_examples.jsonl"

# --- PASSWORD CONFIG ---
LABELING_PASSWORD = os.environ.get("LABELING_PASSWORD", "labeling123")

# --- YOUR NAME CONFIG ---
YOUR_NAME = os.environ.get("YOUR_NAME", "Festus")

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
current_mode = None
current_persona = None
all_messages = []
all_messages_unfiltered = []  # For VLM: all messages including non-image ones for context
current_index = 0
current_conversation = []
labeled_conversations = []
vlm_undo_stack = []  # Track labeled examples for undo functionality


# --- HELPER FUNCTIONS ---
def get_progress_file(persona: str) -> Path:
    """Get persona-specific progress file."""
    if current_mode == "vlm":
        return VLM_LABELED_DIR / f".progress_{persona}.json"
    return LABELED_DIR / f".progress_{persona}.json"

def get_vlm_output_file(persona: str) -> Path:
    """Get persona-specific VLM labeled examples file path."""
    out_dir = VLM_LABELED_DIR / persona
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "labeled_examples.jsonl"


def load_cleaned_messages(persona: str) -> List[Dict]:
    """Load cleaned messages from JSON file."""
    filepath = CLEANED_DIR / f"{persona.upper()}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Cleaned file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        messages = json.load(f)
    
    logger.info(f"Loaded {len(messages)} messages for persona: {persona}")
    return messages


def load_vlm_cleaned_data(persona: str) -> Tuple[List[Dict], Path]:
    """Load cleaned VLM data (messages with images)."""
    persona_dir = VLM_CLEANED_DIR / persona
    data_file = persona_dir / f"{persona}.json"
    
    if not data_file.exists():
        raise FileNotFoundError(f"VLM cleaned data not found: {data_file}")
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_messages = data.get("messages", [])
    
    # Filter to only image messages for VLM labeling
    image_messages = [msg for msg in all_messages if msg.get("image")]
    
    logger.info(f"Loaded {len(all_messages)} total messages, {len(image_messages)} with images for persona: {persona}")
    return image_messages, persona_dir


def load_progress(persona: str) -> Dict:
    """Load labeling progress if it exists."""
    progress_file = get_progress_file(persona)
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "persona": persona,
        "current_index": 0,
        "labeled_conversations": []
    }


def save_progress():
    """Save current labeling progress."""
    progress = {
        "persona": current_persona,
        "current_index": current_index,
    }
    
    # Only save labeled_conversations for core mode, not VLM
    if current_mode != "vlm":
        progress["labeled_conversations"] = labeled_conversations
    
    progress_file = get_progress_file(current_persona)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    
    if current_mode == "vlm":
        logger.info(f"Progress saved for {current_persona}: {current_index}/{len(all_messages)} images")
    else:
        logger.info(f"Progress saved for {current_persona}: {current_index}/{len(all_messages)} messages")


def save_labeled_conversation(conversation: List[Dict], persona: str):
    """
    Save a labeled conversation to JSONL file.
    
    Role assignment:
    - YOUR_NAME is always "assistant"
    - Other person is always "user"
    
    Processing:
    1. Merge consecutive messages from the same sender (standard ChatML format)
    2. Skip conversations where assistant messages first (ChatML requires user to start)
    3. Skip conversations with no assistant messages (you never respond - not training data)
    """
    if len(conversation) < 2:
        return
    
    # Skip conversations where assistant messages first
    if conversation[0]["sender"] == YOUR_NAME:
        logger.debug(f"Skipping conversation: starts with {YOUR_NAME}. ChatML requires user to start.")
        return
    
    # Merge consecutive messages from same sender
    merged_messages = []
    current_role = None
    current_content = []
    
    for msg in conversation:
        role = "assistant" if msg["sender"] == YOUR_NAME else "user"
        
        if role == current_role:
            current_content.append(msg["message"])
        else:
            if current_role is not None:
                merged_messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content)
                })
            current_role = role
            current_content = [msg["message"]]
    
    # Append final message
    if current_role is not None:
        merged_messages.append({
            "role": current_role,
            "content": "\n".join(current_content)
        })
    
    # Skip conversations with no assistant messages
    has_assistant = any(msg["role"] == "assistant" for msg in merged_messages)
    if not has_assistant:
        logger.debug(f"Skipping conversation: no assistant messages.")
        return
    
    # Create conversation object
    conv_obj = {
        "messages": merged_messages,
        "persona": persona.lower(),
        "timestamp_start": conversation[0]["timestamp"],
        "timestamp_end": conversation[-1]["timestamp"]
    }
    
    # Append to JSONL
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(conv_obj, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved conversation with {len(merged_messages)} messages")


def save_vlm_labeled_example(example: Dict, persona: str):
    """Save a labeled VLM training example to JSONL."""
    global vlm_undo_stack
    
    out_path = get_vlm_output_file(persona)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    vlm_undo_stack.append(example)
    logger.info(f"Saved VLM example for {persona}. Undo stack size: {len(vlm_undo_stack)}")


def initialize_labeling(mode: str, persona: str):
    """Initialize labeling session."""
    global current_mode, current_persona, all_messages, all_messages_unfiltered, current_index, current_conversation, labeled_conversations, vlm_undo_stack
    
    current_mode = mode
    current_persona = persona
    
    if mode == "vlm":
        # Load all messages for context, but filter to image messages for labeling
        persona_dir = VLM_CLEANED_DIR / persona
        data_file = persona_dir / f"{persona}.json"
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_messages_unfiltered = data.get("messages", [])
        
        # Normalize roles from sender: persona -> "user", YOUR_NAME -> "assistant"
        for m in all_messages_unfiltered:
            sender = (m.get("sender") or "").strip()
            if sender and sender == YOUR_NAME:
                m["role"] = "assistant"
            else:
                m["role"] = "user"

        all_messages = [msg for msg in all_messages_unfiltered if msg.get("image")]
        logger.info(f"Loaded {len(all_messages_unfiltered)} total messages, {len(all_messages)} with images")
        # Initialize undo stack from existing persona file if present
        vlm_undo_stack = []
        existing_path = get_vlm_output_file(persona)
        if existing_path.exists():
            with open(existing_path, "r", encoding="utf-8") as f:
                vlm_undo_stack = [json.loads(line) for line in f if line.strip()]
    else:
        all_messages = load_cleaned_messages(persona)
        all_messages_unfiltered = []
    
    # Try to resume from progress
    progress = load_progress(persona)
    current_index = progress["current_index"]
    
    # Only load labeled_conversations for core mode
    if mode != "vlm":
        labeled_conversations = progress.get("labeled_conversations", [])
        if current_index > 0 or labeled_conversations:
            logger.info(f"Resumed from index {current_index} with {len(labeled_conversations)} conversations")
        else:
            logger.info("Starting fresh labeling session")
    else:
        if current_index > 0:
            logger.info(f"Resumed VLM labeling from image {current_index}")
        else:
            logger.info("Starting fresh VLM labeling session")
    
    current_conversation = []


# --- AUTHENTICATION HELPER ---
def require_login(f):
    """Decorator to require login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# --- FLASK APP FACTORY ---
def create_app() -> Flask:
    """Create and configure Flask app."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_SORT_KEYS"] = False
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "labeling-secret-key-change-in-production")
    
    # Register routes
    @app.route("/login", methods=["GET", "POST"])
    def login():
        """Login page."""
        if request.method == "POST":
            password = request.form.get("password", "")
            if password == LABELING_PASSWORD:
                session["authenticated"] = True
                logger.info("User authenticated")
                return redirect(url_for("select_mode"))
            else:
                return render_template("login.html", error="Invalid password")
        
        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    def logout():
        """Logout."""
        session.clear()
        logger.info("User logged out")
        return redirect(url_for("login"))

    @app.route("/select-mode", methods=["GET", "POST"])
    @require_login
    def select_mode():
        """Select labelling mode."""
        if request.method == "POST":
            mode = request.form.get("mode", "").strip()
            if mode in ["manual", "vlm"]:
                session["mode"] = mode
                return redirect(url_for("select_persona"))
            return render_template("select_mode.html", error="Invalid mode")
        
        return render_template("select_mode.html")

    @app.route("/select-persona", methods=["GET", "POST"])
    @require_login
    def select_persona():
        """Select persona to label."""
        if request.method == "POST":
            persona = request.form.get("persona", "").strip()
            if not persona:
                return render_template("select_persona.html", error="Persona required")
            
            try:
                mode = session.get("mode", "manual")
                initialize_labeling(mode, persona)
                return redirect(url_for("index"))
            except FileNotFoundError as e:
                return render_template("select_persona.html", error=str(e))
        
        return render_template("select_persona.html")

    @app.route("/")
    @require_login
    def index():
        """Main labeling interface."""
        if not current_persona:
            return redirect(url_for("select_mode"))
        
        if current_mode == "vlm":
            return render_template("vlm_labeler.html", persona=current_persona)
        else:
            return render_template("labeler.html", persona=current_persona, mode=current_mode)

    @app.route("/api/next-message", methods=["GET"])
    @require_login
    def api_next_message():
        """Get the next message to label."""
        if current_index >= len(all_messages):
            return jsonify({
                "done": True,
                "message": "All messages labeled!",
                "total_labeled": len(labeled_conversations)
            })
        
        msg = all_messages[current_index]
        
        return jsonify({
            "done": False,
            "index": current_index,
            "total": len(all_messages),
            "message": {
                "timestamp": msg["timestamp"],
                "sender": msg["sender"],
                "content": msg["message"]
            },
            "conversation_size": len(current_conversation) + 1,
            "labeled_count": len(labeled_conversations)
        })

    @app.route("/api/add-to-conversation", methods=["POST"])
    @require_login
    def api_add_to_conversation():
        """Add current message to conversation."""
        global current_conversation, current_index
        
        if current_index >= len(all_messages):
            return jsonify({"error": "No message to add"}), 400
        
        current_conversation.append(all_messages[current_index])
        current_index += 1
        
        return jsonify({
            "success": True,
            "conversation_size": len(current_conversation),
            "current_index": current_index
        })

    @app.route("/api/end-conversation", methods=["POST"])
    @require_login
    def api_end_conversation():
        """End conversation and save it."""
        global current_conversation, labeled_conversations
        
        if len(current_conversation) == 0:
            return jsonify({"error": "No messages in conversation"}), 400
        
        save_labeled_conversation(current_conversation, current_persona)
        labeled_conversations.append(current_conversation.copy())
        current_conversation = []
        
        save_progress()
        
        return jsonify({
            "success": True,
            "labeled_count": len(labeled_conversations)
        })

    @app.route("/api/undo", methods=["POST"])
    @require_login
    def api_undo():
        """Undo last message addition."""
        global current_index, current_conversation
        
        if len(current_conversation) == 0:
            return jsonify({"error": "Nothing to undo"}), 400
        
        current_conversation.pop()
        current_index -= 1
        
        return jsonify({
            "success": True,
            "current_index": current_index,
            "conversation_size": len(current_conversation)
        })

    @app.route("/api/skip-message", methods=["POST"])
    @require_login
    def api_skip_message():
        """Skip current message."""
        global current_index
        
        if current_index >= len(all_messages):
            return jsonify({"error": "No more messages"}), 400
        
        current_index += 1
        
        return jsonify({
            "success": True,
            "current_index": current_index,
            "total": len(all_messages)
        })

    @app.route("/api/stats", methods=["GET"])
    @require_login
    def api_stats():
        """Get labeling statistics."""
        return jsonify({
            "mode": current_mode,
            "persona": current_persona,
            "total_messages": len(all_messages),
            "labeled_messages": sum(len(conv) for conv in labeled_conversations),
            "labeled_conversations": len(labeled_conversations),
            "current_index": current_index,
            "progress_percent": round((current_index / len(all_messages) * 100) if all_messages else 0, 1)
        })

    @app.route("/api/vlm/image/<persona>/<filename>")
    @require_login
    def api_vlm_image(persona: str, filename: str):
        """Serve VLM image file."""
        from flask import send_file
        image_path = VLM_CLEANED_DIR / persona / "images" / filename
        if not image_path.exists():
            return jsonify({"error": "Image not found"}), 404
        return send_file(image_path, mimetype='image/jpeg')

    @app.route("/api/vlm/context-window", methods=["GET"])
    @require_login
    def api_vlm_context_window():
        """Get context window around current image message with ±20 messages."""
        if current_mode != "vlm" or current_index >= len(all_messages):
            return jsonify({"error": "Invalid request"}), 400
        
        image_msg = all_messages[current_index]
        
        # Find the position of this image message in the unfiltered list
        unfiltered_idx = None
        for idx, msg in enumerate(all_messages_unfiltered):
            if msg.get("image") == image_msg.get("image") and msg.get("timestamp") == image_msg.get("timestamp"):
                unfiltered_idx = idx
                break
        
        if unfiltered_idx is None:
            # Fallback: search by image filename only
            for idx, msg in enumerate(all_messages_unfiltered):
                if msg.get("image") == image_msg.get("image"):
                    unfiltered_idx = idx
                    break
        
        if unfiltered_idx is None:
            # Last resort: use current_index as approximation
            unfiltered_idx = current_index
        
        # Get ±50 messages around the image to ensure at least 20+ available for selection
        context_size = 50
        start_idx = max(0, unfiltered_idx - context_size)
        end_idx = min(len(all_messages_unfiltered), unfiltered_idx + context_size + 1)
        
        preceding = all_messages_unfiltered[start_idx:unfiltered_idx]
        following = all_messages_unfiltered[unfiltered_idx + 1:end_idx]
        
        return jsonify({
            "preceding": preceding,
            "image_message": image_msg,
            "following": following,
            "unfiltered_idx": unfiltered_idx,
            "total_unfiltered": len(all_messages_unfiltered),
            "current_index": current_index,
            "total_images": len(all_messages)
        })

    @app.route("/api/vlm/save-example", methods=["POST"])
    @require_login
    def api_vlm_save_example():
        """Save a VLM training example."""
        global current_index, labeled_conversations
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        example = {
            "images": data.get("images", []),
            "messages": data.get("messages", []),
            "persona": current_persona,
            "timestamp": data.get("timestamp")
        }
        
        save_vlm_labeled_example(example, current_persona)
        labeled_conversations.append(example)
        current_index += 1
        save_progress()
        
        return jsonify({
            "success": True,
            "labeled_count": len(labeled_conversations),
            "current_index": current_index
        })

    @app.route("/api/vlm/skip-image", methods=["POST"])
    @require_login
    def api_vlm_skip_image():
        """Skip current image."""
        global current_index
        
        if current_index >= len(all_messages):
            return jsonify({"error": "No more images"}), 400
        
        current_index += 1
        save_progress()
        
        return jsonify({"success": True})

    @app.route("/api/vlm/undo", methods=["POST"])
    @require_login
    def api_vlm_undo():
        """Undo the last labeled example."""
        global vlm_undo_stack, current_index
        
        if not vlm_undo_stack:
            return jsonify({"error": "Nothing to undo"}), 400
        
        # Remove last example from undo stack
        vlm_undo_stack.pop()
        
        # Rewrite the JSONL file without the last example
        out_path = get_vlm_output_file(current_persona)
        with open(out_path, "w", encoding="utf-8") as f:
            for example in vlm_undo_stack:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        # Move back one image (undo the increment)
        if current_index > 0:
            current_index -= 1
        
        save_progress()
        logger.info(f"Undone last example. Undo stack size: {len(vlm_undo_stack)}")
        
        return jsonify({
            "success": True,
            "undo_stack_size": len(vlm_undo_stack),
            "current_index": current_index
        })

    @app.route("/api/vlm/load-more-before", methods=["GET"])
    @require_login
    def api_vlm_load_more_before():
        """Load more messages before the current context window."""
        if current_mode != "vlm" or current_index >= len(all_messages):
            return jsonify({"error": "Invalid request"}), 400
        
        image_msg = all_messages[current_index]
        
        # Find the position of this image message in the unfiltered list
        unfiltered_idx = None
        for idx, msg in enumerate(all_messages_unfiltered):
            if msg.get("image") == image_msg.get("image") and msg.get("timestamp") == image_msg.get("timestamp"):
                unfiltered_idx = idx
                break
        
        if unfiltered_idx is None:
            for idx, msg in enumerate(all_messages_unfiltered):
                if msg.get("image") == image_msg.get("image"):
                    unfiltered_idx = idx
                    break
        
        if unfiltered_idx is None:
            unfiltered_idx = current_index
        
        # Load 20 more messages before
        context_size = 40  # Expand window
        start_idx = max(0, unfiltered_idx - context_size)
        preceding = all_messages_unfiltered[start_idx:unfiltered_idx]
        
        return jsonify({
            "preceding": preceding,
            "count": len(preceding)
        })

    @app.route("/api/vlm/load-more-after", methods=["GET"])
    @require_login
    def api_vlm_load_more_after():
        """Load more messages after the current context window."""
        if current_mode != "vlm" or current_index >= len(all_messages):
            return jsonify({"error": "Invalid request"}), 400
        
        image_msg = all_messages[current_index]
        
        # Find the position of this image message in the unfiltered list
        unfiltered_idx = None
        for idx, msg in enumerate(all_messages_unfiltered):
            if msg.get("image") == image_msg.get("image") and msg.get("timestamp") == image_msg.get("timestamp"):
                unfiltered_idx = idx
                break
        
        if unfiltered_idx is None:
            for idx, msg in enumerate(all_messages_unfiltered):
                if msg.get("image") == image_msg.get("image"):
                    unfiltered_idx = idx
                    break
        
        if unfiltered_idx is None:
            unfiltered_idx = current_index
        
        # Load 20 more messages after
        context_size = 40  # Expand window
        end_idx = min(len(all_messages_unfiltered), unfiltered_idx + context_size + 1)
        following = all_messages_unfiltered[unfiltered_idx + 1:end_idx]
        
        return jsonify({
            "following": following,
            "count": len(following)
        })
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    logger.info("Use 'make label-core' to start the labeling UI")
