from datasets import load_dataset
import json
import speechcomet
import glob
import os
import argparse
from tqdm import tqdm

def run_eval(args):
    # get checkpoints
    ckpt_dir = os.path.join(args.model_folder, "checkpoints")
    matches = glob.glob(os.path.join(ckpt_dir, "epoch=4-*.ckpt"))
    assert matches, "No checkpoint found for epoch=4"
    checkpoint = matches[0]

    # load model and data
    model = speechcomet.load_from_checkpoint(checkpoint)
    dataset = load_dataset(args.dataset)["dev"]

    outputs = []
    samples = []

    for entry in tqdm(dataset):
        if args.modality == "text":
            sample = {"src": entry["src_text"], "mt": entry["tgt_text"]}
        elif args.modality == "audio":
            raise NotImplementedError
            sample = {"audio": entry["src_audio"], "mt": entry["tgt_text"]}
        elif args.modality == "textaudio":
            raise NotImplementedError
        samples.append(sample)
        entry.pop("audio", None)
        outputs.append(entry)

    scores = model.predict(samples=samples, gpus=1, num_workers=1, batch_size=40).scores

    # save outputs
    with open(f"{args.model_folder}/input_data.jsonl", "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(f"{args.model_folder}/output_scores.jsonl", "w", encoding="utf-8") as f:
        for score in scores:
            f.write(json.dumps(score, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-folder",
        type=str,
        required=True,
        help="Path to Lightning log directory"
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
    run_eval(args)
