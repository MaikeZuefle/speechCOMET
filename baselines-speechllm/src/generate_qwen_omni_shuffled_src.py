import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "..", "..", "evaluation"))

from eval_utils import run_correlation_eval
from generate_qwen_omni import (
    build_conversation_audio,
    build_conversation_audiotext,
    build_conversation_text,
    predict_scores_batch,
)
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


def make_derangement(n, seed=42):
    """Return a length-n permutation array with no fixed points."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not any(perm[i] == i for i in range(n)):
            return perm


def run_eval(args):
    print(f"Loading model: {args.model_name}  (modality: {args.modality})")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    _base = os.path.join(_here, "../..")
    default_out = os.path.join(
        "speechllm-baselines", "results", args.model_name.replace("/", "_")
    )
    output_dir = os.path.join(_base, args.output_name or default_out, "shuffled_src")
    os.makedirs(output_dir, exist_ok=True)

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

    need_audio = args.modality in ("audio", "audiotext", "textaudio")
    convs, tmp_files, all_entries = [], [], []

    for i, entry in enumerate(tqdm(entries, desc="Building conversations")):
        src = entries[src_from[i]]
        mt_text  = entry["tgt_text"]
        src_text = src["src_text"]

        if args.modality == "text":
            convs.append(build_conversation_text(src_text, mt_text))

        elif args.modality == "audio":
            audio_array = src["audio"]["array"]
            sr = src["audio"]["sampling_rate"]
            conv, tmp = build_conversation_audio(audio_array, sr, mt_text)
            convs.append(conv)
            tmp_files.append(tmp)

        elif args.modality in ("audiotext", "textaudio"):
            audio_array = src["audio"]["array"]
            sr = src["audio"]["sampling_rate"]
            conv, tmp = build_conversation_audiotext(audio_array, sr, src_text, mt_text)
            convs.append(conv)
            tmp_files.append(tmp)

        else:
            raise ValueError(f"Unknown modality: {args.modality}")

        out = {k: v for k, v in entry.items() if k != "audio"}
        all_entries.append((f"{entry['src_lang']}-{entry['tgt_lang']}", out))

    scores = []
    for batch_start in tqdm(range(0, len(convs), args.batch_size), desc="Scoring"):
        batch = convs[batch_start:batch_start + args.batch_size]
        scores.extend(predict_scores_batch(model, processor, batch))
    torch.cuda.empty_cache()

    for path in tmp_files:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    grouped_scores: dict[str, list] = defaultdict(list)
    grouped_outputs: dict[str, list] = defaultdict(list)
    for (lp, out), score in zip(all_entries, scores):
        grouped_scores[lp].append(score)
        grouped_outputs[lp].append(out)

    for lp in grouped_scores:
        with open(os.path.join(output_dir, f"input_data_{args.split}_{lp}.jsonl"), "w") as f:
            for item in grouped_outputs[lp]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        score_file = f"output_scores_{args.split}_{lp}_{args.modality}.jsonl"
        with open(os.path.join(output_dir, score_file), "w") as f:
            for s in grouped_scores[lp]:
                f.write(json.dumps(s) + "\n")
        print(f"  Saved {lp}: {len(grouped_scores[lp])} scores")

    # correlation evaluation
    eval_dir = os.path.join(_base, "evaluation", "iwslt26-metrics")
    run_correlation_eval(output_dir, args.split, grouped_scores.keys(), eval_dir, score_suffix=args.modality)

    print(f"Done. Results saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--modality", required=True,
                        choices=["text", "audio", "audiotext", "textaudio"],
                        help="Input modality (use the modality the model was trained on)")
    parser.add_argument("--output-name", default=None,
                        help="Output path relative to repo root "
                             "(default: speechllm-baselines/results/<model-name>)")
    parser.add_argument("--dataset", default="maikezu/iwslt2026-metrics-shared-train-dev")
    parser.add_argument("--split", default="dev", choices=["dev"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_eval(args)
