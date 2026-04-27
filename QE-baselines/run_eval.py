"""
Evaluate QE baselines (asr_comet, asr_comet_partial, blaser, speechqe) on:
  1. dev / dev_asr  — segment scores + correlation + WER analysis
  2. mustshe        — pairwise accuracy
  3. contraprost    — pairwise accuracy

Usage (run from repo root):
    python QE-baselines/run_eval.py --method asr_comet  --task dev_asr
    python QE-baselines/run_eval.py --method asr_comet  --task mustshe
    python QE-baselines/run_eval.py --method blaser     --task contraprost
    python QE-baselines/run_eval.py --method speechqe   --task dev_asr \\
        --speechqe-model-de h-j-han/SpeechQE-TowerInstruct-7B-en2de
"""
import argparse
import os
import sys
import json
import subprocess
from collections import defaultdict
from tqdm import tqdm

import pandas as pd

# Allow imports from evaluation/ (eval_utils, mustshe_eval, contraprost_eval)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))

from eval_utils import load_contraprost_csv_files, run_correlation_eval
import mustshe_eval as _mustshe
import contraprost_eval as _contraprost

# ─── Method metadata ────────────────────────────────────────────────────────

METHODS = {
    "asr_comet":         {"modality": "text",  "output_dir": "QE-baselines/results/qe-comet"},
    "asr_comet_partial": {"modality": "text",  "output_dir": "QE-baselines/results/qe-comet-partial"},
    "blaser":            {"modality": "audio", "output_dir": "QE-baselines/results/qe-blaser"},
    "speechqe":          {"modality": "audio", "output_dir": "QE-baselines/results/qe-speechqe"},
}

_LANG_NAMES = {
    "de": "German", "zh": "Chinese", "es": "Spanish",
    "fr": "French",  "it": "Italian", "ja": "Japanese",
}

def _speechqe_lang_config(lang):
    name = _LANG_NAMES.get(lang, lang)
    return {
        "inst":          f"Given the {name} translation of the speech, estimate the quality of the translation as a score between 0 to 1.",
        "suffix_format": f"\n{name} translation: {{x}}",
    }

# ─── Scorer loaders ─────────────────────────────────────────────────────────

def load_comet_scorer(model_name):
    import comet
    model = comet.load_from_checkpoint(comet.download_model(model_name))
    def score(rows):
        return model.predict([{"src": r["src"], "mt": r["mt"]} for r in rows]).scores
    return score


def load_blaser_scorer():
    import torch
    import librosa
    from sonar.models.blaser.loader import load_blaser_model
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blaser      = load_blaser_model("blaser_2_0_qe").eval().to(device)
    text_emb    = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=device)
    speech_emb  = SpeechToEmbeddingModelPipeline(
        encoder="sonar_speech_encoder_eng", device=device)
    langcode    = {"en": "eng_Latn", "de": "deu_Latn", "zh": "zho_Hans", "es": "spa_Latn", "fr": "fra_Latn", "it": "ita_Latn", "ja": "jpn_Jpan"}

    def score(rows):
        import tempfile
        import soundfile as sf
        import numpy as np
        unknown = {r["tgt_lang"] for r in rows if r["tgt_lang"] not in langcode}
        if unknown:
            raise ValueError(f"BLASER: no SONAR langcode for tgt_lang(s): {unknown}. Add them to the langcode dict.")
        scores = []
        for r in tqdm(rows, desc="BLASER scoring", leave=False):
            if r.get("audio_array") is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                sf.write(tmp_path, np.array(r["audio_array"]), r["audio_sr"])
                try:
                    wav, _ = librosa.load(tmp_path, sr=16000, mono=True)
                finally:
                    os.remove(tmp_path)
            else:
                wav, _ = librosa.load(r["audio_path"], sr=16000, mono=True)
            wvf = torch.tensor(wav).unsqueeze(0).to(device)
            src_emb = speech_emb.predict([wvf])
            mt_emb  = text_emb.predict([r["mt"]], source_lang=langcode[r["tgt_lang"]])
            scores.append(blaser(src=src_emb, mt=mt_emb).item())
        return scores
    return score


def load_speechqe_scorer(models_by_lang, chunk_size=500):
    """
    models_by_lang: dict tgt_lang -> HF model name or local path
    Runs score_speechqe.py via subprocess; groups rows by tgt_lang.
    """
    import tempfile

    speechqe_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SpeechQE")

    def score(rows):
        import soundfile as sf
        import numpy as np

        # Validate upfront: every lang must have a model (or fall back to "de")
        fallback_model = models_by_lang.get("de")
        missing_models = {r.get("tgt_lang", "de") for r in rows
                         if r.get("tgt_lang", "de") not in models_by_lang and not fallback_model}
        if missing_models:
            raise ValueError(
                f"No SpeechQE model configured for tgt_lang(s): {missing_models} "
                f"and no 'de' fallback available.")

        # Group rows by model (not by language) to load each model only once.
        # Languages without an explicit model fall back to the "de" model (zero-shot).
        zero_shot_langs = {r.get("tgt_lang", "de") for r in rows
                           if r.get("tgt_lang", "de") not in models_by_lang}
        if zero_shot_langs:
            print(f"  WARNING: no SpeechQE model for {sorted(zero_shot_langs)}, using de model as zero-shot fallback")
        model_groups = defaultdict(list)  # model -> [(original_index, row)]
        for i, r in enumerate(rows):
            lang = r.get("tgt_lang", "de")
            model = models_by_lang.get(lang, fallback_model)
            model_groups[model].append((i, r))

        scores = [None] * len(rows)

        for model, indexed_rows in model_groups.items():
            indices    = [i for i, _ in indexed_rows]
            model_rows = [r for _, r in indexed_rows]
            model_scores = []

            chunks = [
                (indices[c:c+chunk_size], model_rows[c:c+chunk_size])
                for c in range(0, len(model_rows), chunk_size)
            ]
            print(f"  SpeechQE: {len(model_rows)} rows → {len(chunks)} chunk(s) of ≤{chunk_size}")

            for chunk_idx, (_, chunk_rows) in enumerate(chunks):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tsv_data = []
                    for j, r in enumerate(chunk_rows):
                        lang = r.get("tgt_lang", "de")
                        cfg  = _speechqe_lang_config(lang)
                        if r.get("audio_array") is not None:
                            wav_path = os.path.join(tmpdir, f"audio_{j}.wav")
                            sf.write(wav_path, np.array(r["audio_array"]), r["audio_sr"])
                        else:
                            wav_path = r["audio_path"]
                        tsv_data.append({
                            "path":     wav_path,
                            "sentence": "",
                            "split":    "eval",
                            "lang":     r.get("src_lang", "en"),
                            "task":     f"qe.{r.get('src_lang','en')}2{lang}",
                            "inst":     cfg["inst"],
                            "suffix":   cfg["suffix_format"].format(x=r["mt"]),
                        })

                    manifest = "eval.tsv"
                    pd.DataFrame(tsv_data).to_csv(
                        os.path.join(tmpdir, manifest), sep="\t", index=False)

                    output_name = f"speechqe_eval_chunk{chunk_idx}"
                    env = os.environ.copy()
                    env["PYTHONPATH"] = speechqe_dir + os.pathsep + env.get("PYTHONPATH", "")
                    subprocess.run(
                        [
                            "python", "speechqe/score_speechqe.py",
                            f"--dataroot={tmpdir}",
                            f"--manifest_files={manifest}",
                            f"--speechqe_model={model}",
                            f"--output_file_name={output_name}",
                            "--speech_from_file_path=True",
                        ],
                        cwd=speechqe_dir,
                        env=env,
                        check=True,
                    )

                    output_json = os.path.join(speechqe_dir, "outputs", f"{output_name}.json")
                    with open(output_json) as f:
                        data = json.load(f)
                    model_scores.extend(data["pred_strings_float"])

            for idx, s in zip(indices, model_scores):
                scores[idx] = s

        if any(s is None for s in scores):
            raise RuntimeError("Some rows did not receive a SpeechQE score.")
        return scores

    return score


def get_scorer(method, args=None):
    if method == "asr_comet":
        return load_comet_scorer("Unbabel/wmt22-cometkiwi-da")
    elif method == "asr_comet_partial":
        return load_comet_scorer("zouharvi/COMET-partial")
    elif method == "blaser":
        return load_blaser_scorer()
    elif method == "speechqe":
        models_by_lang = {}
        if args and args.speechqe_model_de:
            models_by_lang["de"] = args.speechqe_model_de
        if args and args.speechqe_model_zh:
            models_by_lang["zh"] = args.speechqe_model_zh
        if not models_by_lang:
            raise ValueError("speechqe requires at least one of --speechqe-model-de or --speechqe-model-zh")
        chunk_size = getattr(args, "speechqe_chunk_size", 500) or 500
        return load_speechqe_scorer(models_by_lang, chunk_size=chunk_size)
    else:
        raise ValueError(f"Unknown method: {method}")


# ─── Task: dev / dev_asr ────────────────────────────────────────────────────

def _decode_hf_audio(item):
    """Decode an HF audio value to (numpy float32 array, sample_rate).
    Handles both already-decoded dicts and lazy AudioDecoder objects."""
    import io
    import soundfile as sf
    import numpy as np
    # Already decoded by HF datasets
    if isinstance(item, dict):
        return np.array(item["array"], dtype="float32"), int(item["sampling_rate"])
    # Lazy AudioDecoder — read from encoded bytes or path
    encoded = getattr(item, "_hf_encoded", None)
    if encoded is not None and encoded.get("bytes"):
        data, sr = sf.read(io.BytesIO(encoded["bytes"]), dtype="float32", always_2d=False)
    elif encoded is not None and encoded.get("path"):
        data, sr = sf.read(encoded["path"], dtype="float32", always_2d=False)
    else:
        raise ValueError(
            f"Cannot decode audio: unrecognised type {type(item)}. "
            f"attrs: {[a for a in dir(item) if not a.startswith('__')]}")
    return np.array(data, dtype="float32"), int(sr)


def run_dev(args, scorer, output_dir, modality):
    import tempfile
    import soundfile as sf
    from datasets import load_dataset
    dataset = load_dataset("maikezu/scottish-metrics")[args.split]

    # Write audio to temp WAV files immediately so arrays don't stay in RAM
    # while the scorer subprocess loads a large model.
    audio_tmpdir = tempfile.TemporaryDirectory() if modality == "audio" else None

    all_rows, all_lang_pairs = [], []
    for idx, entry in enumerate(tqdm(dataset, desc="Building samples")):
        lang_pair = f"{entry['src_lang']}-{entry['tgt_lang']}"
        orig_audio_path = entry.get("audio_path", "")
        if modality == "audio":
            audio_array, audio_sr = _decode_hf_audio(entry["audio"])
            wav_path = os.path.join(audio_tmpdir.name, f"audio_{idx}.wav")
            sf.write(wav_path, audio_array, audio_sr)
            del audio_array  # free immediately; WAV on disk is the reference
            scorer_audio_path = wav_path
            row_audio_sr      = audio_sr
        else:
            scorer_audio_path = orig_audio_path
            row_audio_sr      = 16000
        row = {
            "audio_path":      scorer_audio_path,  # temp WAV for scorer
            "orig_audio_path": orig_audio_path,     # original path for WER lookup
            "doc_id":          entry.get("doc_id", ""),
            "src_text":        entry.get("src_text", ""),
            "src_text_system": entry.get("src_text_system", ""),
            "src_lang":        entry.get("src_lang", ""),
            "tgt_lang":        entry.get("tgt_lang", ""),
            "domain":          entry.get("domain", ""),
            "tgt_system":      entry.get("tgt_system", ""),
            "tgt_text":        entry.get("tgt_text", ""),
            "score":           float(entry.get("score", 0)),
            # scorer fields
            "src":             entry.get("src_text", ""),
            "mt":              entry.get("tgt_text", ""),
            # audio_array is intentionally None: audio lives in wav files above
            "audio_array":     None,
            "audio_sr":        row_audio_sr,
        }
        all_rows.append(row)
        all_lang_pairs.append(lang_pair)

    # Free the HF dataset (Arrow mmap + decoded audio cache) before the scorer
    # subprocess loads a large model — both live in the same 60 GB SLURM alloc.
    del dataset
    import gc; gc.collect()

    print(f"Scoring {len(all_rows)} samples with {args.method}...")
    try:
        scores = scorer(all_rows)
    finally:
        if audio_tmpdir is not None:
            audio_tmpdir.cleanup()

    grouped_scores  = defaultdict(list)
    grouped_outputs = defaultdict(list)
    for lang_pair, row, score in zip(all_lang_pairs, all_rows, scores):
        grouped_scores[lang_pair].append(score)
        grouped_outputs[lang_pair].append(row)

    os.makedirs(output_dir, exist_ok=True)
    for lang_pair in grouped_scores:
        input_path  = os.path.join(output_dir, f"input_data_{args.split}_{lang_pair}.jsonl")
        scores_path = os.path.join(output_dir, f"output_scores_{args.split}_{lang_pair}.jsonl")
        with open(input_path, "w") as f:
            for row in grouped_outputs[lang_pair]:
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
            for s in grouped_scores[lang_pair]:
                f.write(f"{s}\n")
        print(f"  Saved {lang_pair}: {len(grouped_scores[lang_pair])} scores → {scores_path}")

    # correlation evaluation
    eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation", "iwslt26-metrics"))
    run_correlation_eval(output_dir, args.split, grouped_scores.keys(), eval_dir)

    # WER correlation analysis
    if args.split == "dev_asr" and args.wer_csv:
        for thresh in [80, 90]:
            subprocess.run([
                "python", "evaluation/wer_correlation_analysis.py",
                "--model-dir", output_dir,
                "--split", args.split,
                "--wer-csv", args.wer_csv,
                "--challenge-score-threshold", str(thresh),
            ], check=True)


# ─── Task: MuST-SHE ─────────────────────────────────────────────────────────

def run_mustshe(args, scorer, output_dir, modality):
    mustshe_dir = args.mustshe_dir
    df = _mustshe.load_csv_files(mustshe_dir)

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "src":        row.get("src", ""),
            "mt":         row["mt"],
            "audio_path": row["audio_path"],
            "src_lang":   row.get("src_lang", "en"),
            "tgt_lang":   row.get("lang", "de"),
        })

    print(f"Scoring {len(rows)} MuST-SHE samples with {args.method}...")
    scores = scorer(rows)
    df = df.copy()
    df["model_score"] = scores

    results = _mustshe.compute_results(df)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "mustshe_scores.csv"), index=False)
    results.to_csv(os.path.join(output_dir, "mustshe_results.csv"), index=False)
    _mustshe.print_mustshe_pivot(results)
    print(f"Saved to {output_dir}/mustshe_results.csv")


# ─── Task: ContraProST ───────────────────────────────────────────────────────

def run_contraprost(args, scorer, output_dir, modality):
    df = load_contraprost_csv_files(args.contraprost_dir)

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "src":        row.get("src", ""),
            "mt":         row["mt"],
            "audio_path": row["src_audio"],
            "src_lang":   row.get("src_lang", "en"),
            "tgt_lang":   row["lang"],
        })

    print(f"Scoring {len(rows)} ContraProST samples with {args.method}...")
    scores = scorer(rows)
    df = df.copy()
    df["model_score"] = scores

    results = _contraprost.compute_results(df)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "contraprost_scores.csv"), index=False)
    results.to_csv(os.path.join(output_dir, "contraprost_results.csv"), index=False)
    _contraprost.print_contraprost_results(results)
    print(f"Saved to {output_dir}/contraprost_results.csv")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()))
    parser.add_argument("--task", required=True,
                        choices=["dev", "dev_asr", "mustshe", "contraprost"])
    parser.add_argument("--split", default=None,
                        help="Dataset split (inferred from --task if omitted)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: trained_models/qe-<method>)")
    parser.add_argument("--mustshe-dir",
                        default="data/MuST-SHE_v1.2/MuST-SHE-v1.2-data/tsv")
    parser.add_argument("--contraprost-dir", default="data/contraProST")
    # SpeechQE language-pair-specific models
    parser.add_argument("--speechqe-model-de", default=None,
                        help="SpeechQE model for en→de (e.g. h-j-han/SpeechQE-TowerInstruct-7B-en2de)")
    parser.add_argument("--speechqe-model-zh", default=None,
                        help="SpeechQE model for en→zh")
    parser.add_argument("--speechqe-chunk-size", type=int, default=6000,
                        help="Samples per SpeechQE subprocess call (reduce if OOM)")
    parser.add_argument("--speechqe-python", default=None,
                        help="Python executable for SpeechQE subprocess "
                             "(default: same interpreter running this script). "
                             "Override if SpeechQE deps are in a different env.")
    parser.add_argument("--wer-csv", default=None,
                        help="Path to WER CSV for WER correlation analysis (dev_asr only)")
    args = parser.parse_args()

    # Infer split from task
    if args.split is None:
        args.split = args.task if args.task in ("dev", "dev_asr") else "dev_asr"

    meta = METHODS[args.method]
    output_dir = args.output_dir or meta["output_dir"]
    modality   = meta["modality"]

    print(f"Method: {args.method}  |  Task: {args.task}  |  Output: {output_dir}")
    scorer = get_scorer(args.method, args)

    if args.task in ("dev", "dev_asr"):
        run_dev(args, scorer, output_dir, modality)
    elif args.task == "mustshe":
        run_mustshe(args, scorer, output_dir, modality)
    elif args.task == "contraprost":
        run_contraprost(args, scorer, output_dir, modality)


if __name__ == "__main__":
    main()
