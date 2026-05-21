"""
SpeechCOMET inference with shuffled source inputs.

For each example the source (audio / text) is replaced with the source
from a randomly chosen *different* example in the same language pair
(deterministic derangement, default seed=42).  The target text and
human score are kept from the original example.

For audiotext models, --shuffle-modality controls which source is shuffled:
  both  (default): shuffle audio AND text together (same random partner)
  audio: shuffle audio only, keep real text source
  text:  shuffle text only, keep real audio source

Results are saved to:
  shuffled_src/   (--shuffle-modality both, default)
  shuffled_audio/ (--shuffle-modality audio)
  shuffled_text/  (--shuffle-modality text)

Usage (run from repo root):
    python evaluation/run_inf_shuffled_src.py \\
        --model-folder trained_models/harris-20ep \\
        --modality audio \\
        --split dev_asr

    python evaluation/run_inf_shuffled_src.py \\
        --model-folder trained_models/orkney-sum-from-text-ckpt-20ep \\
        --modality audiotext \\
        --split dev \\
        --shuffle-modality audio
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import speechcomet
from datasets import load_dataset
from speechcomet import download_model
from tqdm import tqdm


def make_derangement(n, seed=42):
    """Return a length-n permutation array with no fixed points."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not any(perm[i] == i for i in range(n)):
            return perm


def run_eval(args):
    if args.hf_model:
        model = speechcomet.load_from_checkpoint(download_model(args.hf_model))
        base_dir = args.hf_model.replace("/", "_")
    else:
        ckpt_dir = os.path.join(args.model_folder, "checkpoints")
        matches = glob.glob(os.path.join(ckpt_dir, "epoch=*-*.ckpt"))
        if not matches:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        checkpoint = max(
            matches,
            key=lambda p: int(os.path.basename(p).split("epoch=")[1].split("-")[0]),
        )
        print(f"Loading checkpoint: {checkpoint}")
        model = speechcomet.load_from_checkpoint(checkpoint)
        base_dir = args.model_folder

    subdir = {"both": "shuffled_src", "audio": "shuffled_audio", "text": "shuffled_text"}[args.shuffle_modality]
    output_dir = os.path.join(base_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {args.dataset}[{args.split}] ...")
    entries = list(load_dataset(args.dataset)[args.split])

    # Build per-lang-pair derangement: src_from[i] = index whose source to use for example i
    lp_indices: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        lp_indices[f"{e['src_lang']}-{e['tgt_lang']}"].append(i)

    src_from: dict[int, int] = {}
    for indices in lp_indices.values():
        perm = make_derangement(len(indices), seed=args.seed)
        for local_i, global_i in enumerate(indices):
            src_from[global_i] = indices[perm[local_i]]

    all_samples, all_outputs, all_lang_pairs = [], [], []
    for i, entry in enumerate(tqdm(entries, desc="Building samples")):
        src = entries[src_from[i]]
        lang_pair = f"{entry['src_lang']}-{entry['tgt_lang']}"

        if args.modality == "text":
            sample = {"src": src["src_text"], "mt": entry["tgt_text"]}
        elif args.modality == "audio":
            sample = {"src_audio": src["audio"], "mt": entry["tgt_text"]}
        elif args.modality in ("textaudio", "audiotext"):
            if args.shuffle_modality == "both":
                sample = {"src_audio": src["audio"], "src": src["src_text"], "mt": entry["tgt_text"]}
            elif args.shuffle_modality == "audio":
                # shuffle audio only, keep real text
                sample = {"src_audio": src["audio"], "src": entry["src_text"], "mt": entry["tgt_text"]}
            elif args.shuffle_modality == "text":
                # shuffle text only, keep real audio
                sample = {"src_audio": entry["audio"], "src": src["src_text"], "mt": entry["tgt_text"]}
        else:
            raise ValueError(f"Unknown modality: {args.modality}")

        # Output keeps original target metadata; audio_path is the original (for WER join)
        out = {k: v for k, v in entry.items() if k != "audio"}
        all_samples.append(sample)
        all_outputs.append(out)
        all_lang_pairs.append(lang_pair)

    print(f"Running inference on {len(all_samples)} samples ...")
    all_scores = model.predict(
        samples=all_samples, gpus=1, num_workers=1, batch_size=args.batch_size
    ).scores

    grouped_scores: dict[str, list] = defaultdict(list)
    grouped_outputs: dict[str, list] = defaultdict(list)
    for lp, out, score in zip(all_lang_pairs, all_outputs, all_scores):
        grouped_scores[lp].append(score)
        grouped_outputs[lp].append(out)

    for lp in grouped_scores:
        with open(os.path.join(output_dir, f"input_data_{args.split}_{lp}.jsonl"), "w") as f:
            for item in grouped_outputs[lp]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(os.path.join(output_dir, f"output_scores_{args.split}_{lp}.jsonl"), "w") as f:
            for s in grouped_scores[lp]:
                f.write(json.dumps(s) + "\n")
        print(f"  Saved {lp}: {len(grouped_scores[lp])} scores")

    print(f"Done. Results saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", default=None,
                        help="Path to Lightning log directory")
    parser.add_argument("--hf-model", default=None,
                        help="HuggingFace model repo id")
    parser.add_argument("--dataset", default="maikezu/scottish-metrics")
    parser.add_argument("--modality", required=True,
                        choices=["text", "audio", "audiotext", "textaudio"])
    parser.add_argument("--split", default="dev_asr", choices=["dev", "dev_asr"])
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-modality", default="both",
                        choices=["both", "audio", "text"],
                        help="Which source modality to shuffle (audiotext models only). "
                             "'both' shuffles audio+text together (default); "
                             "'audio' shuffles audio while keeping real text; "
                             "'text' shuffles text while keeping real audio.")
    args = parser.parse_args()
    if not args.hf_model and not args.model_folder:
        parser.error("Either --hf-model or --model-folder must be provided")
    if args.shuffle_modality != "both" and args.modality not in ("audiotext", "textaudio"):
        parser.error("--shuffle-modality audio/text only applies to audiotext models")
    run_eval(args)
