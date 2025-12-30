import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import requests


# Paths (relative to repo root inferred from this file location)
REPO_ROOT = Path(__file__).parents[2]
DATASETS_DIR = REPO_ROOT / "datasets" / "core"
CLEANED_DIR = DATASETS_DIR / "chats_cleaned"
EVALS_DIR = DATASETS_DIR / "chats_evals" / "llm_system_scan"
EVALS_DIR.mkdir(parents=True, exist_ok=True)

# Heuristic patterns for WhatsApp system/media artifacts and export leftovers
SYSTEM_PATTERNS = [
    "end-to-end encrypted",
    "joined using this phone number",
    "this message was deleted",
    "you deleted this message",
    "missed voice call",
    "missed video call",
    "changed their phone number",
    "left",
    "added",
    "removed",
    "voice call",
    "is a contact.",
]
MEDIA_PATTERNS = [
    "<media omitted>",
    "omitted",
    "image",
    "audio",
    "sticker",
    "gif",
    "document",
    "video",
]
# Timestamps or header-like artifacts that may slip through if parsing missed a line
EXPORT_HEADER_PATTERNS = [
    r"\[?\d{1,2}/\d{1,2}/\d{2,4}[,\s]+\d{1,2}:\d{2}(:\d{2})?\]?",  # [DD/MM/YYYY, HH:MM(:SS)]
    r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*-\s*",            # DD/MM/YYYY, HH:MM -
]

# Simple PII residue checks (should be masked by cleaning; use as a safety net)
PII_REGEX = {
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",
    "PHONE": r"\+?\d{1,3}[\s-]?(?:\(\d+\))?[\s-]?\d{3,}[\s-]?\d{3,}",
}


def load_cleaned(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pattern_scan(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        text = str(m.get("message", ""))
        lower = text.lower()
        reasons: List[str] = []
        if any(p in lower for p in SYSTEM_PATTERNS):
            reasons.append("whatsapp_system_phrase")
        if any(p in lower for p in MEDIA_PATTERNS):
            reasons.append("media_artifact")
        if any(re.search(p, text) for p in EXPORT_HEADER_PATTERNS):
            reasons.append("timestamp_or_export_header")
        for name, rx in PII_REGEX.items():
            if re.search(rx, text, flags=re.IGNORECASE):
                reasons.append(f"pii_residue:{name}")
        if reasons:
            findings.append({"index": i, "reason": ",".join(reasons), "snippet": text[:200]})
    return findings


def chunk(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def build_prompt(batch: List[Tuple[int, str]]) -> List[Dict[str, str]]:
    schema = (
        "Return strict JSON with keys: findings (list of objects with index and reason), "
        "and summary (object with counts by category)."
    )
    guidance = (
        "You are auditing cleaned WhatsApp chats to detect residual system or export artifacts that should not be in training data. "
        "Flag messages that look like: system notifications (left/added/removed/number changed), encryption banners, deleted message notices, call logs, media placeholders, timestamps or export headers, or other metadata. "
        "Be conservative: only flag when the content is clearly an artifact."
    )
    examples = (
        "Examples to flag: 'Missed voice call', 'end-to-end encrypted', '<Media omitted>', 'You deleted this message', "
        "lines that are just timestamps like '[12/01/2023, 10:22:11]'."
    )
    numbered = "\n".join([f"{idx}: {text}" for idx, text in batch])
    user = (
        f"{guidance}\n{examples}\n{schema}\nMessages (index: text):\n{numbered}\n"
        "Only output JSON with keys findings and summary."
    )
    return [
        {"role": "system", "content": "You are an expert data hygiene auditor."},
        {"role": "user", "content": user},
    ]


def call_openai_chat(model: str, messages: List[Dict[str, str]], timeout: int = 60) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "openai package is required. Install with: pip install openai"
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment")

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        timeout=timeout,
    )
    return resp.choices[0].message.content or ""


def call_ollama_chat(model: str, messages: List[Dict[str, str]], base_url: str, timeout: int = 60) -> str:
    """Call an Ollama chat model via REST API and return assistant text content.

    Expects an Ollama server (default http://localhost:11434). Uses /api/chat with stream=false.
    """
    if not base_url:
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # Keep deterministic behavior similar to temperature=0
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # When stream=false, Ollama returns a single object with message.content
        msg = data.get("message", {})
        content = msg.get("content")
        if not content and isinstance(data, dict):
            # Some versions return .messages list
            messages_obj = data.get("messages")
            if isinstance(messages_obj, list) and messages_obj:
                content = messages_obj[-1].get("content")
        if not content:
            raise ValueError("No content field in Ollama response")
        return content
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def parse_json_block(text: str) -> Dict[str, Any]:
    # Try raw JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try fenced code blocks
    m = re.search(r"```(?:json)?\n([\s\S]*?)\n```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Last resort: find first '{' to last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    raise ValueError("Could not parse JSON from model output")


def llm_scan(
    messages: List[Dict[str, Any]],
    provider: str,
    model: str,
    base_url: str,
    batch_size: int,
    timeout: int,
    ppm_delay: float,
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    pairs: List[Tuple[int, str]] = [(i, str(m.get("message", ""))) for i, m in enumerate(messages)]
    for batch in chunk(pairs, batch_size):
        prompt = build_prompt(batch)
        if provider == "openai":
            out = call_openai_chat(model, prompt, timeout=timeout)
        elif provider == "ollama":
            out = call_ollama_chat(model, prompt, base_url=base_url, timeout=timeout)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        data = parse_json_block(out)
        for f in data.get("findings", []):
            idx = int(f.get("index"))
            reason = str(f.get("reason", "llm_flag"))
            snippet = messages[idx].get("message", "")[:200]
            findings.append({"index": idx, "reason": reason, "snippet": snippet})
        # light rate limiting between calls
        if ppm_delay > 0:
            time.sleep(ppm_delay)
    return findings


def merge_findings(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = {(f["index"], f["reason"]) for f in a}
    for f in b:
        key = (f["index"], f["reason"])
        if key not in seen:
            a.append(f)
            seen.add(key)
    return a


def scan_file(
    path: Path,
    provider: str,
    model: str,
    base_url: str,
    max_messages: int,
    batch_size: int,
    timeout: int,
    ppm_delay: float,
    pattern_only: bool,
) -> Dict[str, Any]:
    messages = load_cleaned(path)
    if max_messages > 0:
        messages = messages[:max_messages]
    persona = path.stem

    deterministic = pattern_scan(messages)
    all_findings = list(deterministic)

    if not pattern_only and model:
        try:
            llm_findings = llm_scan(
                messages,
                provider=provider,
                model=model,
                base_url=base_url,
                batch_size=batch_size,
                timeout=timeout,
                ppm_delay=ppm_delay,
            )
            all_findings = merge_findings(all_findings, llm_findings)
        except Exception as e:
            print(f"[WARN] LLM scan failed for {path.name}: {e}", file=sys.stderr)

    all_findings.sort(key=lambda x: x["index"])  # stable ordering

    summary: Dict[str, int] = {}
    for f in all_findings:
        for tag in str(f["reason"]).split(","):
            summary[tag] = summary.get(tag, 0) + 1

    return {
        "persona": persona,
        "file": str(path),
        "total_messages": len(messages),
        "findings": all_findings,
        "summary": summary,
    }


def save_report(report: Dict[str, Any]) -> Path:
    persona = report.get("persona", "report")
    out_path = EVALS_DIR / f"{persona}.llm_system_scan.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Scan cleaned chats with an external LLM (Ollama or OpenAI) to flag residual WhatsApp system data.")
    p.add_argument("--input-dir", type=str, default=str(CLEANED_DIR), help="Directory of cleaned chat JSONs")
    p.add_argument("--provider", type=str, choices=["ollama", "openai"], default="ollama", help="LLM provider to use")
    p.add_argument("--model", type=str, default="llama3", help="Model name (Ollama model name or OpenAI model id)")
    p.add_argument("--ollama-base-url", type=str, default=os.getenv("OLLAMA_HOST", "http://localhost:11434"), help="Ollama server base URL")
    p.add_argument("--max-messages-per-file", type=int, default=0, help="Limit messages per file (0 = all)")
    p.add_argument("--batch-size", type=int, default=50, help="Messages per LLM call")
    p.add_argument("--timeout", type=int, default=60, help="Per-call timeout seconds")
    p.add_argument("--ppm-delay", type=float, default=0.4, help="Sleep between calls (seconds)")
    p.add_argument("--pattern-only", action="store_true", help="Skip LLM calls; run deterministic scan only")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        print(f"Input dir not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    reports: Dict[str, Any] = {
        "scanned_at": int(time.time()),
        "provider": None if args.pattern_only else args.provider,
        "model": None if args.pattern_only else args.model,
        "files": [],
    }

    for path in sorted(in_dir.glob("*.json")):
        print(f"Scanning {path.name} ...")
        report = scan_file(
            path=path,
            provider=args.provider,
            model=None if args.pattern_only else args.model,
            base_url=args.ollama_base_url,
            max_messages=args.max_messages_per_file,
            batch_size=args.batch_size,
            timeout=args.timeout,
            ppm_delay=args.ppm_delay,
            pattern_only=args.pattern_only,
        )
        reports["files"].append(report)

        out_path = save_report(report)
        print(f"Report -> {out_path}")

    # Also write an aggregate index
    index_path = EVALS_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"Index -> {index_path}")


if __name__ == "__main__":
    main()
