"""
QE-baseline evaluation with shuffled source inputs.

For each example the source (audio / text) is replaced with the source
from a randomly chosen *different* example in the same language pair
(deterministic derangement, default seed=42).  The target text and
human score are kept from the original example.

Results are saved to <method_output_dir>/shuffled_src/.

Usage (run from repo root):
    python baselines-QE/baseline_eval_shuffled.py --method asr_comet  --split dev
    python baselines-QE/baseline_eval_shuffled.py --method blaser     --split dev
    python baselines-QE/baseline_eval_shuffled.py --method speechqe   --split dev \\
        --speechqe-model-de h-j-han/SpeechQE-TowerInstruct-7B-en2de
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_eval import METHODS, _decode_hf_audio, get_scorer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "speechcomet-eval"))
from eval_utils import run_correlation_eval


def make_derangement(n, seed=42):
    """Return a length-n permutation array with no fixed points."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not any(perm[i] == i for i in range(n)):
            return perm


def run(args):
    meta = METHODS[args.method]
    modality = meta["modality"]
    output_dir = os.path.join(args.output_dir or meta["output_dir"], "shuffled_src")
    os.makedirs(output_dir, exist_ok=True)

    scorer = get_scorer(args.method, args)

    print(f"Loading {args.dataset}[{args.split}] ...")
    entries = list(load_dataset(args.dataset)[args.split])

    # Build per-lang-pair derangement
    lp_indices: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        lp_indices[f"{e['src_lang']}-{e['tgt_lang']}"].append(i)

    src_from: dict[int, int] = {}
    for indices in lp_indices.values():
        perm = make_derangement(len(indices), seed=args.seed)
        for local_i, global_i in enumerate(indices):
            src_from[global_i] = indices[perm[local_i]]

    audio_tmpdir = tempfile.TemporaryDirectory() if modality == "audio" else None

    all_rows, all_lang_pairs = [], []
    for i, entry in enumerate(tqdm(entries, desc="Building samples")):
        src = entries[src_from[i]]
        lang_pair = f"{entry['src_lang']}-{entry['tgt_lang']}"
        orig_audio_path = entry.get("audio_path", "")

        if modality == "audio":
            audio_array, audio_sr = _decode_hf_audio(src["audio"])
            wav_path = os.path.join(audio_tmpdir.name, f"audio_{i}.wav")
            sf.write(wav_path, audio_array, audio_sr)
            del audio_array
            scorer_audio_path = wav_path
        else:
            scorer_audio_path = orig_audio_path
            audio_sr = 16000

        row = {
            # scorer fields
            "audio_path":      scorer_audio_path,
            "orig_audio_path": orig_audio_path,      # original audio path (for WER join)
            "src":             src.get("src_text", ""),
            "mt":              entry.get("tgt_text", ""),
            "audio_array":     None,
            "audio_sr":        audio_sr,
            # metadata (target stays original)
            "doc_id":          entry.get("doc_id", ""),
            "src_text":        src.get("src_text", ""),
            "src_text_system": src.get("src_text_system", ""),
            "src_lang":        entry.get("src_lang", ""),
            "tgt_lang":        entry.get("tgt_lang", ""),
            "domain":          entry.get("domain", ""),
            "tgt_system":      entry.get("tgt_system", ""),
            "tgt_text":        entry.get("tgt_text", ""),
            "score":           float(entry.get("score", 0)),
        }
        all_rows.append(row)
        all_lang_pairs.append(lang_pair)

    print(f"Scoring {len(all_rows)} samples with {args.method} (shuffled src) ...")
    try:
        scores = scorer(all_rows)
    finally:
        if audio_tmpdir is not None:
            audio_tmpdir.cleanup()

    grouped_scores: dict[str, list] = defaultdict(list)
    grouped_outputs: dict[str, list] = defaultdict(list)
    for lp, row, score in zip(all_lang_pairs, all_rows, scores):
        grouped_scores[lp].append(score)
        grouped_outputs[lp].append(row)

    for lp in grouped_scores:
        input_path  = os.path.join(output_dir, f"input_data_{args.split}_{lp}.jsonl")
        scores_path = os.path.join(output_dir, f"output_scores_{args.split}_{lp}.jsonl")
        with open(input_path, "w") as f:
            for row in grouped_outputs[lp]:
                row_out = {
                    "audio_path":      row["orig_audio_path"],
                    "doc_id":          row["doc_id"],
                    "src_text":        row["src_text"],
                    "src_text_system": row["src_text_system"],
                    "src_lang":        row["src_lang"],
                    "tgt_lang":        row["tgt_lang"],
                    "domain":          row["domain"],
                    "tgt_system":      row["tgt_system"],
                    "tgt_text":        row["tgt_text"],
                    "score":           row["score"],
                }
                f.write(json.dumps(row_out) + "\n")
        with open(scores_path, "w") as f:
            for s in grouped_scores[lp]:
                f.write(f"{s}\n")
        print(f"  Saved {lp}: {len(grouped_scores[lp])} scores → {scores_path}")

    # correlation evaluation
    eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "speechcomet-eval", "iwslt26-metrics"))
    run_correlation_eval(output_dir, args.split, grouped_scores.keys(), eval_dir)

    print(f"Done. Results in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["asr_comet", "asr_comet_partial", "blaser", "speechqe"])
    parser.add_argument("--split", default="dev", choices=["dev"])
    parser.add_argument("--dataset", default="maikezu/iwslt2026-metrics-shared-train-dev",
                        help="HuggingFace dataset repo to load")
    parser.add_argument("--output-dir", default=None,
                        help="Override base output dir (shuffled_src/ appended automatically)")
    parser.add_argument("--speechqe-model-de", default=None)
    parser.add_argument("--speechqe-model-zh", default=None)
    parser.add_argument("--speechqe-chunk-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)
