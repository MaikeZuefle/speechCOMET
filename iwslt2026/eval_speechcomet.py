"""
SpeechCOMET evaluation on the IWSLT 2026 test set.

Handles mixed audio: ACL domain uses HF AudioDecoder (already segmented),
other domains extract segments from long WAV/MP4 files using timestamps.

Usage (from repo root):
    python iwslt2026/eval_speechcomet.py \\
        --model-folder trained_models/orkney-sum-from-text-ckpt-BIG \\
        --modality audiotext \\
        --audio-base-dir /path/to/iwslt2026data
"""
import argparse
import glob
import json
import os
import re
import sys
import tempfile
from collections import defaultdict

import speechcomet
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from speechcomet import download_model
from iwslt2026.audio_utils import get_src_audio


def load_model(args):
    if args.hf_model:
        model = speechcomet.load_from_checkpoint(download_model(args.hf_model))
        output_dir = args.hf_model.replace("/", "_")
    else:
        ckpt_dir = os.path.join(args.model_folder, "checkpoints")
        matches = [
            p for p in glob.glob(os.path.join(ckpt_dir, "epoch=*-val_kendall=*.ckpt"))
            if not os.path.basename(p).startswith("worse_")
            and re.search(r"val_kendall=(\d+\.\d+)", os.path.basename(p))
        ]
        if not matches:
            raise FileNotFoundError(f"No val_kendall checkpoints found in {ckpt_dir}")
        checkpoint = max(
            matches,
            key=lambda p: float(re.search(r"val_kendall=(\d+\.\d+)", os.path.basename(p)).group(1))
        )
        print(f"Loading checkpoint: {checkpoint}")
        model = speechcomet.load_from_checkpoint(checkpoint)
        output_dir = args.model_folder
    return model, output_dir


def run_eval(args):
    model, output_dir = load_model(args)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {args.dataset}[{args.split}] ...")
    dataset = load_dataset(args.dataset)[args.split]

    if args.limit:
        n = len(dataset)
        # select() is Arrow-backed (fast); non-ACL at the start, ACL toward the end
        head = dataset.select(range(min(args.limit, n)))
        tail = dataset.select(range(max(0, n - args.limit * 20), n))
        non_acl = [e for e in head if e["audio"] is None][:args.limit]
        acl     = [e for e in tail if e["audio"] is not None][:args.limit]
        if not acl:
            raise RuntimeError("--limit: no ACL entries found in the last segment of the dataset")
        dataset = non_acl + acl
        print(f"  Limiting to {len(acl)} ACL + {len(non_acl)} non-ACL entries")

    all_samples = []
    all_outputs = []
    all_lang_pairs = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Building samples (temp audio dir: {tmp_dir}) ...")
        for entry in tqdm(dataset):
            lang_pair = f"{entry['src_lang']}-{entry['tgt_lang']}"

            if args.modality == "text":
                sample = {"src": entry["src_text"], "mt": entry["tgt_text"]}
            elif args.modality == "audio":
                src_audio = get_src_audio(entry, args.audio_base_dir, tmp_dir)
                sample = {"src_audio": src_audio, "mt": entry["tgt_text"]}
            elif args.modality in ("audiotext", "textaudio"):
                src_audio = get_src_audio(entry, args.audio_base_dir, tmp_dir)
                sample = {"src_audio": src_audio, "src": entry["src_text"], "mt": entry["tgt_text"]}
            else:
                raise ValueError(f"Unknown modality: {args.modality}")

            all_samples.append(sample)
            all_lang_pairs.append(lang_pair)
            out = {k: v for k, v in entry.items() if k not in ("audio",)}
            all_outputs.append(out)

        print(f"Running inference on {len(all_samples)} samples ...")
        all_scores = model.predict(
            samples=all_samples, gpus=1, num_workers=1, batch_size=args.batch_size
        ).scores

    # Group by lang pair and save
    grouped_scores = defaultdict(list)
    grouped_outputs = defaultdict(list)
    for lang_pair, output, score in zip(all_lang_pairs, all_outputs, all_scores):
        grouped_scores[lang_pair].append(score)
        grouped_outputs[lang_pair].append(output)

    for lang_pair in grouped_scores:
        input_path = os.path.join(output_dir, f"input_data_{args.split}_{lang_pair}.jsonl")
        scores_path = os.path.join(output_dir, f"output_scores_{args.split}_{lang_pair}.jsonl")
        with open(input_path, "w", encoding="utf-8") as f:
            for item in grouped_outputs[lang_pair]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(scores_path, "w", encoding="utf-8") as f:
            for score in grouped_scores[lang_pair]:
                f.write(json.dumps(score, ensure_ascii=False) + "\n")
        print(f"  Saved {lang_pair}: {len(grouped_scores[lang_pair])} scores")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", default=None)
    parser.add_argument("--hf-model", default=None,
                        help="HuggingFace model repo id")
    parser.add_argument("--dataset", default="maikezu/iwslt2026-metrics-shared-test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--modality", required=True,
                        choices=["text", "audio", "audiotext", "textaudio"])
    parser.add_argument("--audio-base-dir", default=None,
                        help="Base directory for non-ACL audio files "
                             "(required for audio/audiotext modalities)")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None,
                        help="Quick-test mode: take only this many ACL + this many non-ACL entries")
    args = parser.parse_args()

    if not args.hf_model and not args.model_folder:
        parser.error("Either --hf-model or --model-folder must be provided")
    if args.modality != "text" and args.audio_base_dir is None:
        parser.error("--audio-base-dir is required for audio/audiotext modalities")

    run_eval(args)
