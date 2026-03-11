from datasets import load_dataset
from speechcomet import download_model
import json
import speechcomet
import glob
import os
import argparse
from collections import defaultdict
from tqdm import tqdm


def run_eval(args):
    # load model
    if args.hf_model:
        model = speechcomet.load_from_checkpoint(download_model(args.hf_model))
        output_dir = args.hf_model.replace("/", "_")
        os.makedirs(output_dir, exist_ok=True)
    else:
        ckpt_dir = os.path.join(args.model_folder, "checkpoints")
        matches = glob.glob(os.path.join(ckpt_dir, "epoch=*-*.ckpt"))
        checkpoint = max(
            matches,
            key=lambda p: int(os.path.basename(p).split("epoch=")[1].split("-")[0])
        )
        model = speechcomet.load_from_checkpoint(checkpoint)
        output_dir = args.model_folder

    dataset = load_dataset(args.dataset)["dev"]

    # Collect all samples, preserving lang pair info for splitting later
    all_samples = []
    all_outputs = []
    all_lang_pairs = []

    for entry in tqdm(dataset):
        lang_pair = f"{entry['src_lang']}-{entry['tgt_lang']}"

        if args.modality == "text":
            sample = {"src": entry["src_text"], "mt": entry["tgt_text"]}
        elif args.modality == "audio":
            sample = {"src_audio": entry["audio"], "mt": entry["tgt_text"]}
        elif args.modality in ("textaudio", "audiotext"):
            sample = {"src_audio": entry["audio"], "src": entry["src_text"], "mt": entry["tgt_text"]}

        all_samples.append(sample)
        all_lang_pairs.append(lang_pair)
        entry.pop("audio", None)
        all_outputs.append(entry)


    all_scores = model.predict(samples=all_samples, gpus=1, num_workers=1, batch_size=40).scores

    # Split scores and outputs by lang pair and save
    grouped_scores = defaultdict(list)
    grouped_outputs = defaultdict(list)
    for lang_pair, output, score in zip(all_lang_pairs, all_outputs, all_scores):
        grouped_scores[lang_pair].append(score)
        grouped_outputs[lang_pair].append(output)

    for lang_pair in grouped_scores:
        with open(f"{output_dir}/input_data_{lang_pair}.jsonl", "w", encoding="utf-8") as f:
            for item in grouped_outputs[lang_pair]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(f"{output_dir}/output_scores_{lang_pair}.jsonl", "w", encoding="utf-8") as f:
            for score in grouped_scores[lang_pair]:
                f.write(json.dumps(score, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-folder",
        type=str,
        default=None,
        help="Path to Lightning log directory"
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="HuggingFace model repo id (e.g. maikezu/shetland)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        help="text or audio or audiotext"
    )
    args = parser.parse_args()
    if not args.hf_model and not args.model_folder:
        parser.error("Either --hf-model or --model-folder must be provided")
    run_eval(args)