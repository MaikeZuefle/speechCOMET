from datasets import load_dataset
from speechcomet import download_model
import json
import speechcomet
import glob
import os
import argparse
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
    outputs = []
    samples = []
    for entry in tqdm(dataset):
        if args.modality == "text":
            sample = {"src": entry["src_text"], "mt": entry["tgt_text"]}
        elif args.modality == "audio":
            sample = {"src_audio": entry["audio"], "mt": entry["tgt_text"]}
        elif args.modality == "textaudio" or "audiotext":
            sample = {"src_audio": entry["audio"], "src": entry["src_text"], "mt": entry["tgt_text"]}
        samples.append(sample)
        entry.pop("audio", None)
        outputs.append(entry)

    scores = model.predict(samples=samples, gpus=1, num_workers=1, batch_size=40).scores

    with open(f"{output_dir}/input_data.jsonl", "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(f"{output_dir}/output_scores.jsonl", "w", encoding="utf-8") as f:
        for score in scores:
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