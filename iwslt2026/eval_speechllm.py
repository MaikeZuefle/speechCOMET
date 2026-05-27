"""
SpeechLLM (Qwen2.5-Omni) evaluation on the IWSLT 2026 test set (audiotext modality).

Handles mixed audio: ACL domain uses HF AudioDecoder (already segmented),
other domains extract segments from long WAV/MP4 files using timestamps.

Usage (from repo root):
    python iwslt2026/eval_speechllm.py \\
        --model-name baselines-speechllm/saves/qwen2.5-omni-7b/merged/iwslt26_textaudio \\
        --output-name baselines-speechllm/results/Qwen_Qwen2.5-Omni-7B-iwslt26-textaudio \\
        --audio-base-dir iwslt2026data
"""
import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_utils import _resolve_audio_path, _extract_segment_ffmpeg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines-speechllm", "src"))
from generate_qwen_omni import build_conversation_audiotext, predict_scores_batch
from prompts import get_prompt


def get_audio_array(entry, audio_base_dir, tmp_dir):
    """Return (audio_array, sample_rate) for a test set entry.

    - ACL: decode from AudioDecoder (already segmented), save to temp WAV, reload
    - Others: extract segment from long file using timestamps
    """
    if entry["audio"] is not None:
        audio = entry["audio"]
        if isinstance(audio, dict):
            # HF already-decoded dict: {"array": np.ndarray, "sampling_rate": int}
            audio_data = np.array(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
        else:
            # Lazy AudioDecoder — call get_all_samples()
            decoded = audio.get_all_samples()
            audio_data = decoded.data.squeeze(0).numpy()
            sr = int(decoded.sample_rate)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=0)
        doc_id = entry.get("doc_id", "unknown").replace("/", "_")
        tmp_wav = os.path.join(tmp_dir, f"acl_{doc_id}.wav")
        sf.write(tmp_wav, audio_data.astype(np.float32), sr)
        audio_array, sr = librosa.load(tmp_wav, sr=None, mono=True)
        return audio_array, sr

    # Non-ACL: extract segment via ffmpeg
    full_path = _resolve_audio_path(entry["audio_path"], audio_base_dir)
    start = entry["start_timestamp"]
    end = entry["end_timestamp"]
    doc_id = entry.get("doc_id", "unknown").replace("/", "_")
    tgt_lang = entry.get("tgt_lang", "xx")
    out_path = os.path.join(tmp_dir, f"{doc_id}_{tgt_lang}_{start:.3f}_{end:.3f}.wav")

    if not os.path.exists(out_path):
        _extract_segment_ffmpeg(full_path, start, end, out_path, sr=16000)

    audio_array, sr = librosa.load(out_path, sr=None, mono=True)
    return audio_array, sr


def run_eval(args):
    sp = get_prompt(args.prompt or "standard")

    print(f"Loading model: {args.model_name}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    _base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    output_dir = os.path.join(_base, args.output_name)
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

    all_entries = []
    audiotext_convs = []
    tmp_files = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for entry in tqdm(dataset, desc="Building conversations"):
            src_text = entry["src_text"]
            mt_text = entry["tgt_text"]
            lang_pair = f"{entry['src_lang']}-{entry['tgt_lang']}"

            audio_array, sr = get_audio_array(entry, args.audio_base_dir, tmp_dir)
            conv, tmp_wav = build_conversation_audiotext(audio_array, sr, src_text, mt_text, system_prompt=sp)
            audiotext_convs.append(conv)
            tmp_files.append(tmp_wav)

            output = {k: v for k, v in entry.items() if k != "audio"}
            all_entries.append((lang_pair, output))

        BATCH_SIZE = 1
        scores = []
        for i in tqdm(range(0, len(audiotext_convs), BATCH_SIZE), desc="Scoring audiotext"):
            batch = audiotext_convs[i:i + BATCH_SIZE]
            scores.extend(predict_scores_batch(model, processor, batch))
            torch.cuda.empty_cache()

    for path in tmp_files:
        if os.path.exists(path):
            os.unlink(path)

    grouped_scores = defaultdict(list)
    grouped_outputs = defaultdict(list)
    for (lang_pair, output), score in zip(all_entries, scores):
        grouped_scores[lang_pair].append(score)
        grouped_outputs[lang_pair].append(output)

    for lang_pair in grouped_scores:
        input_path = os.path.join(output_dir, f"input_data_{args.split}_{lang_pair}.jsonl")
        scores_path = os.path.join(output_dir, f"output_scores_{args.split}_{lang_pair}_audiotext.jsonl")
        with open(input_path, "w", encoding="utf-8") as f:
            for item in grouped_outputs[lang_pair]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(scores_path, "w", encoding="utf-8") as f:
            for s in grouped_scores[lang_pair]:
                f.write(json.dumps(s) + "\n")
        print(f"  Saved {lang_pair}: {len(grouped_scores[lang_pair])} scores → {scores_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", default="maikezu/iwslt2026-metrics-shared-test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-name", required=True,
                        help="Output directory path relative to repo root")
    parser.add_argument("--audio-base-dir", required=True,
                        help="Base directory for non-ACL audio files")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Quick-test mode: take only this many ACL + this many non-ACL entries")
    args = parser.parse_args()
    run_eval(args)
