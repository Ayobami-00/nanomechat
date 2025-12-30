#!/usr/bin/env python3

import argparse
import json
import os
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class EvalSplitConfig:
    input_dir: Path
    output_dir: Path
    train_dir: Path
    seed: int = 42
    eval_ratio: float = 0.1
    min_eval_per_persona: int = 25
    max_eval_per_persona: int | None = None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def discover_persona_files(chats_processed_dir: Path) -> Dict[str, Path]:
    persona_files: Dict[str, Path] = {}
    for p in sorted(chats_processed_dir.glob("*.jsonl")):
        if p.name == "conversations.jsonl":
            continue
        persona_files[p.stem] = p
    return persona_files


def split_eval(rows: List[Dict[str, Any]], *, rng: random.Random, cfg: EvalSplitConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        return [], []

    indices = list(range(len(rows)))
    rng.shuffle(indices)

    desired = int(round(len(rows) * cfg.eval_ratio))
    desired = max(desired, cfg.min_eval_per_persona)
    desired = min(desired, len(rows))
    if cfg.max_eval_per_persona is not None:
        desired = min(desired, cfg.max_eval_per_persona)

    eval_idx = set(indices[:desired])
    eval_rows = [rows[i] for i in range(len(rows)) if i in eval_idx]
    remaining_rows = [rows[i] for i in range(len(rows)) if i not in eval_idx]
    return eval_rows, remaining_rows


def build_eval_datasets(cfg: EvalSplitConfig) -> Dict[str, Any]:
    rng = random.Random(cfg.seed)

    persona_files = discover_persona_files(cfg.input_dir)
    if not persona_files:
        raise FileNotFoundError(
            f"No persona jsonl files found in {cfg.input_dir}. Expected files like persona_1.jsonl, persona_2.jsonl, persona_3.jsonl"
        )

    sft_eval_all: List[Dict[str, Any]] = []
    sft_train_all: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "input_dir": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
        "train_dir": str(cfg.train_dir),
        "seed": cfg.seed,
        "eval_ratio": cfg.eval_ratio,
        "min_eval_per_persona": cfg.min_eval_per_persona,
        "max_eval_per_persona": cfg.max_eval_per_persona,
        "personas": {},
    }

    for persona, path in persona_files.items():
        rows = list(_iter_jsonl(path))

        # For determinism per persona, fork RNG with a stable seed derived from base seed + persona.
        seed_material = f"{cfg.seed}:{persona}".encode("utf-8")
        persona_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        persona_rng = random.Random(persona_seed)
        eval_rows, remaining_rows = split_eval(rows, rng=persona_rng, cfg=cfg)

        # LoRA eval: per-persona eval set (adapter-specific)
        lora_out = cfg.output_dir / "lora" / f"{persona}.jsonl"
        _write_jsonl(lora_out, eval_rows)

        # LoRA train: per-persona training set (adapter-specific)
        lora_train_out = cfg.train_dir / "lora" / f"{persona}.jsonl"
        _write_jsonl(lora_train_out, remaining_rows)

        # SFT eval: combined eval set across personas (single-model evaluation)
        sft_eval_all.extend(eval_rows)

        # SFT train: combined training set across personas
        sft_train_all.extend(remaining_rows)

        summary["personas"][persona] = {
            "source_file": str(path),
            "total": len(rows),
            "eval": len(eval_rows),
            "remaining": len(remaining_rows),
            "lora_eval_path": str(lora_out),
            "lora_train_path": str(lora_train_out),
        }

    # Write combined SFT eval file
    sft_out = cfg.output_dir / "sft" / "conversations_eval.jsonl"
    # Shuffle combined eval for better mixing
    rng.shuffle(sft_eval_all)
    _write_jsonl(sft_out, sft_eval_all)

    # Write combined SFT training file
    sft_train_out = cfg.train_dir / "sft" / "conversations_train.jsonl"
    rng.shuffle(sft_train_all)
    _write_jsonl(sft_train_out, sft_train_all)
    summary["sft"] = {
        "eval_total": len(sft_eval_all),
        "sft_eval_path": str(sft_out),
        "train_total": len(sft_train_all),
        "sft_train_path": str(sft_train_out),
    }

    # Placeholder folders for later stages (VLM/GRPO) so the pipeline layout is stable
    (cfg.output_dir / "vlm").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "grpo").mkdir(parents=True, exist_ok=True)

    (cfg.train_dir / "vlm").mkdir(parents=True, exist_ok=True)
    (cfg.train_dir / "grpo").mkdir(parents=True, exist_ok=True)

    meta_path = cfg.output_dir / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> EvalSplitConfig:
    parser = argparse.ArgumentParser(
        description="Create evaluation datasets from datasets/core/chats_processed/*.jsonl"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/core/chats_processed",
        help="Directory containing persona-specific ChatML jsonl files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/core/chats_evals",
        help="Directory to write evaluation datasets",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="datasets/core/chats_train",
        help="Directory to write training datasets (remaining split)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Fraction of each persona file to allocate to eval",
    )
    parser.add_argument(
        "--min_eval_per_persona",
        type=int,
        default=25,
        help="Minimum eval conversations per persona",
    )
    parser.add_argument(
        "--max_eval_per_persona",
        type=int,
        default=None,
        help="Optional cap for eval conversations per persona",
    )

    args = parser.parse_args()

    return EvalSplitConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        train_dir=Path(args.train_dir),
        seed=args.seed,
        eval_ratio=args.eval_ratio,
        min_eval_per_persona=args.min_eval_per_persona,
        max_eval_per_persona=args.max_eval_per_persona,
    )


def main() -> None:
    cfg = parse_args()

    # Defensive checks (helpful error messages)
    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {cfg.input_dir}")

    summary = build_eval_datasets(cfg)
    print("\nEval datasets created successfully.")
    print(f"Output: {cfg.output_dir}")
    print(f"Train: {cfg.train_dir}")
    print("Personas:")
    for persona, stats in summary.get("personas", {}).items():
        print(f"- {persona}: eval={stats['eval']} remaining={stats['remaining']} total={stats['total']}")
    print(f"SFT eval total: {summary['sft']['eval_total']}")
    print(f"SFT train total: {summary['sft']['train_total']}")


if __name__ == "__main__":
    main()
