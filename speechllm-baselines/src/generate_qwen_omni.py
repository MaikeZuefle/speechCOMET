from datasets import load_dataset
import json
import os
import argparse
from collections import defaultdict
from tqdm import tqdm
import torch
import librosa
import soundfile as sf
import tempfile
import re

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from utils import (
    load_mustshe_csv_files, check_missing_audio,
    compute_mustshe_results, print_mustshe_pivot,
    load_contraprost_csv_files, compute_contraprost_results, print_contraprost_results,
)

SYSTEM_PROMPT = "You are an evaluator. Given the source and/or audio and a translation, respond with only a single float score between 0 and 1 indicating translation quality. Output nothing else."

def build_conversation_text(src_text, mt_text):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"Source: {src_text}\nTranslation: {mt_text}"}
        ]},
    ]

def build_conversation_audio(audio_array, sr, mt_text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sr)
        tmp_path = tmp.name
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": tmp_path},
            {"type": "text", "text": f"Translation: {mt_text}"}
        ]},
    ], tmp_path

def build_conversation_audiotext(audio_array, sr, src_text, mt_text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_array, sr)
        tmp_path = tmp.name
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": tmp_path},
            {"type": "text", "text": f"Source: {src_text}\nTranslation: {mt_text}"}
        ]},
    ], tmp_path


def predict_scores_batch(model, processor, conversations_batch):
    """Process a batch of conversations at once, returns list of floats."""
    texts = [
        processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        for conv in conversations_batch
    ]

    audios, images, videos = process_mm_info(conversations_batch, use_audio_in_video=False)

    inputs = processor(
        text=texts,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False
    )
    inputs = inputs.to(model.device).to(model.dtype)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        text_ids = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=16, return_audio=False)

    text_ids = text_ids[:, input_len:]
    decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    scores = []
    for raw in decoded:
        raw = raw.strip()
        try:
            scores.append(float(raw))
        except ValueError:
            match = re.search(r"\d+\.?\d*", raw)
            scores.append(float(match.group()) if match else 0.0)
    return scores


def run_eval(args):
    print(f"Loading model: {args.model_name}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    _base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    output_dir = os.path.join(_base, args.output_name)
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset)[args.split]

    grouped_scores = defaultdict(list)
    grouped_outputs = defaultdict(list)

    all_entries = []
    text_convs = []
    audio_convs = []
    audiotext_convs = []
    tmp_files = []

    for entry in tqdm(dataset, desc="Building conversations"):
        src_text = entry["src_text"]
        mt_text = entry["tgt_text"]
        audio_array = entry["audio"]["array"]
        sr = entry["audio"]["sampling_rate"]

        text_convs.append(build_conversation_text(src_text, mt_text))

        audio_conv, tmp1 = build_conversation_audio(audio_array, sr, mt_text)
        audio_convs.append(audio_conv)
        tmp_files.append(tmp1)

        audiotext_conv, tmp2 = build_conversation_audiotext(audio_array, sr, src_text, mt_text)
        audiotext_convs.append(audiotext_conv)
        tmp_files.append(tmp2)

        lang_pair = f"{entry['src_lang']}-{entry['tgt_lang']}"
        output = {k: v for k, v in entry.items() if k != "audio"}
        all_entries.append((lang_pair, output))

    BATCH_SIZE = 1

    def run_batched(convs, desc):
        scores = []
        for batch_start in tqdm(range(0, len(convs), BATCH_SIZE), desc=desc):
            batch = convs[batch_start:batch_start + BATCH_SIZE]
            scores.extend(predict_scores_batch(model, processor, batch))
        return scores

    text_scores      = run_batched(text_convs,      "Scoring text")
    torch.cuda.empty_cache()
    audio_scores     = run_batched(audio_convs,     "Scoring audio")
    torch.cuda.empty_cache()
    audiotext_scores = run_batched(audiotext_convs, "Scoring audiotext")
    torch.cuda.empty_cache()

    for path in tmp_files:
        os.unlink(path)

    for (lang_pair, output), ts, as_, ats in zip(all_entries, text_scores, audio_scores, audiotext_scores):
        grouped_scores[lang_pair].append({"text": ts, "audio": as_, "audiotext": ats})
        grouped_outputs[lang_pair].append(output)

    for lang_pair in grouped_scores:
        with open(f"{output_dir}/input_data_{args.split}_{lang_pair}.jsonl", "w", encoding="utf-8") as f:
            for item in grouped_outputs[lang_pair]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(f"{output_dir}/output_scores_{args.split}_{lang_pair}_text.jsonl", "w", encoding="utf-8") as f:
            for score in grouped_scores[lang_pair]:
                f.write(json.dumps(score["text"], ensure_ascii=False) + "\n")

        with open(f"{output_dir}/output_scores_{args.split}_{lang_pair}_audio.jsonl", "w", encoding="utf-8") as f:
            for score in grouped_scores[lang_pair]:
                f.write(json.dumps(score["audio"], ensure_ascii=False) + "\n")

        with open(f"{output_dir}/output_scores_{args.split}_{lang_pair}_audiotext.jsonl", "w", encoding="utf-8") as f:
            for score in grouped_scores[lang_pair]:
                f.write(json.dumps(score["audiotext"], ensure_ascii=False) + "\n")

    print(f"Done. Results saved to {output_dir}/")


def run_mustshe(args):
    print(f"Loading model: {args.model_name}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    _base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    output_dir = os.path.join(_base, args.output_name, "mustshe")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading MuST-SHE CSV files from {args.mustshe_dir}")
    df = load_mustshe_csv_files(args.mustshe_dir)
    df = check_missing_audio(
        df, missing_txt="../data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/missing.txt"
    )
    # Drop rows with missing audio (warned above)
    df = df[df["audio_path"].apply(os.path.exists)].reset_index(drop=True)

    text_convs, audio_convs, audiotext_convs, tmp_files = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building conversations"):
        audio_array, sr = librosa.load(row["audio_path"], sr=None, mono=True)

        text_convs.append(build_conversation_text(row["src"], row["mt"]))

        audio_conv, tmp1 = build_conversation_audio(audio_array, sr, row["mt"])
        audio_convs.append(audio_conv)
        tmp_files.append(tmp1)

        audiotext_conv, tmp2 = build_conversation_audiotext(audio_array, sr, row["src"], row["mt"])
        audiotext_convs.append(audiotext_conv)
        tmp_files.append(tmp2)

    BATCH_SIZE = 1

    def run_batched(convs, desc):
        scores = []
        for i in tqdm(range(0, len(convs), BATCH_SIZE), desc=desc):
            scores.extend(predict_scores_batch(model, processor, convs[i:i + BATCH_SIZE]))
        return scores

    text_scores      = run_batched(text_convs,      "Scoring text")
    torch.cuda.empty_cache()
    audio_scores     = run_batched(audio_convs,     "Scoring audio")
    torch.cuda.empty_cache()
    audiotext_scores = run_batched(audiotext_convs, "Scoring audiotext")
    torch.cuda.empty_cache()

    for path in tmp_files:
        os.unlink(path)

    df = df.copy()
    df["score_text"]      = text_scores
    df["score_audio"]     = audio_scores
    df["score_audiotext"] = audiotext_scores

    scores_path = os.path.join(output_dir, "mustshe_scores.csv")
    df.to_csv(scores_path, index=False)
    print(f"\nSaved raw scores to {scores_path}")

    for modality, col in [("text", "score_text"), ("audio", "score_audio"), ("audiotext", "score_audiotext")]:
        results = compute_mustshe_results(df, score_col=col)
        print_mustshe_pivot(results, modality=modality)
        results.to_csv(os.path.join(output_dir, f"mustshe_results_{modality}.csv"), index=False)


def run_contraprost(args):
    print(f"Loading model: {args.model_name}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    _base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    output_dir = os.path.join(_base, args.output_name, "contraprost")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading ContraProST CSV files from {args.contraprost_dir}")
    df = load_contraprost_csv_files(args.contraprost_dir)

    text_convs, audio_convs, audiotext_convs, tmp_files = [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building conversations"):
        audio_array, sr = librosa.load(row["audio_path"], sr=None, mono=True)

        text_convs.append(build_conversation_text(row["src"], row["mt"]))

        audio_conv, tmp1 = build_conversation_audio(audio_array, sr, row["mt"])
        audio_convs.append(audio_conv)
        tmp_files.append(tmp1)

        audiotext_conv, tmp2 = build_conversation_audiotext(audio_array, sr, row["src"], row["mt"])
        audiotext_convs.append(audiotext_conv)
        tmp_files.append(tmp2)

    BATCH_SIZE = 1

    def run_batched(convs, desc):
        scores = []
        for i in tqdm(range(0, len(convs), BATCH_SIZE), desc=desc):
            scores.extend(predict_scores_batch(model, processor, convs[i:i + BATCH_SIZE]))
        return scores

    text_scores      = run_batched(text_convs,      "Scoring text")
    torch.cuda.empty_cache()
    audio_scores     = run_batched(audio_convs,     "Scoring audio")
    torch.cuda.empty_cache()
    audiotext_scores = run_batched(audiotext_convs, "Scoring audiotext")
    torch.cuda.empty_cache()

    for path in tmp_files:
        os.unlink(path)

    df = df.copy()
    df["score_text"]      = text_scores
    df["score_audio"]     = audio_scores
    df["score_audiotext"] = audiotext_scores

    scores_path = os.path.join(output_dir, "contraprost_scores.csv")
    df.to_csv(scores_path, index=False)
    print(f"\nSaved raw scores to {scores_path}")

    for modality, col in [("text", "score_text"), ("audio", "score_audio"), ("audiotext", "score_audiotext")]:
        results = compute_contraprost_results(df, score_col=col)
        print_contraprost_results(results, modality=modality)
        results.to_csv(os.path.join(output_dir, f"contraprost_results_{modality}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Omni-7B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Output subdirectory name (defaults to model-name with / replaced by _)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (for standard eval)")
    parser.add_argument("--split", type=str, default="dev_asr",
                        choices=["dev", "dev_asr"],
                        help="Dataset split to evaluate on (default: dev_asr)")
    parser.add_argument("--mustshe-dir", type=str, default=None,
                        help="Path to MuST-SHE-v1.2-data/tsv/ for MuST-SHE eval")
    parser.add_argument("--contraprost-dir", type=str, default=None,
                        help="Path to contraProST directory containing en_*_expanded.csv files")
    args = parser.parse_args()
    if args.output_name is None:
        args.output_name = args.model_name.replace("/", "_")

    if args.mustshe_dir:
        run_mustshe(args)
    elif args.contraprost_dir:
        run_contraprost(args)
    elif args.dataset:
        run_eval(args)
    else:
        parser.error("One of --dataset, --mustshe-dir, or --contraprost-dir must be provided")