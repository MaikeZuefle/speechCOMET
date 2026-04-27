"""Shared utilities for SpeechCOMET contrastive evaluation scripts."""
import glob
import os
import re
import subprocess

import pandas as pd


def run_correlation_eval(output_dir, split, lang_pairs, eval_dir, score_suffix=""):
    """Run iwslt26-metrics correlation evaluation, print results, and save to .txt files.

    Args:
        output_dir:    directory containing input_data and output_scores files
        split:         dataset split, e.g. "dev_asr"
        lang_pairs:    iterable of lang pairs to evaluate, e.g. ["en-de", "en-zh"]
        eval_dir:      absolute path to evaluation/iwslt26-metrics/
        score_suffix:  optional suffix on score filename, e.g. "text" for SpeechLLM
    """
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"Evaluation dir not found: {eval_dir}")
    suffix = f"_{score_suffix}" if score_suffix else ""
    for lp in lang_pairs:
        scores_file = os.path.abspath(os.path.join(output_dir, f"output_scores_{split}_{lp}{suffix}.jsonl"))
        input_file  = os.path.abspath(os.path.join(output_dir, f"input_data_{split}_{lp}.jsonl"))
        result = subprocess.run(
            ["python", "evaluation/__main__.py", "-i", input_file, "-m", scores_file],
            cwd=eval_dir, check=True, capture_output=True, text=True,
        )
        print(result.stdout)
        corr_path = os.path.join(output_dir, f"correlation_{split}_{lp}.txt")
        with open(corr_path, "w") as f:
            f.write(result.stdout)
        print(f"  Correlation results saved to {corr_path}")



def load_model(model_folder=None, hf_model=None):
    import speechcomet
    from speechcomet import download_model
    if hf_model:
        model = speechcomet.load_from_checkpoint(download_model(hf_model))
        output_dir = hf_model.replace("/", "_")
    else:
        ckpt_dir = os.path.join(model_folder, "checkpoints")
        matches = glob.glob(os.path.join(ckpt_dir, "epoch=*-*.ckpt"))
        checkpoint = max(
            matches,
            key=lambda p: int(os.path.basename(p).split("epoch=")[1].split("-")[0])
        )
        print(f"Loading checkpoint: {checkpoint}")
        model = speechcomet.load_from_checkpoint(checkpoint)
        output_dir = model_folder
    return model, output_dir


def build_sample(row, modality, audio_col="audio_path"):
    if modality == "audio":
        return {"src_audio": row[audio_col], "mt": row["mt"]}
    elif modality == "text":
        return {"src": row["src"], "mt": row["mt"]}
    elif modality in ("textaudio", "audiotext"):
        return {"src_audio": row[audio_col], "src": row["src"], "mt": row["mt"]}
    else:
        raise ValueError(f"Unknown modality: {modality}")


def run_inference(df, model, batch_size, modality, audio_col="audio_path"):
    if modality != "text":
        missing_mask = ~df[audio_col].apply(os.path.exists)
        if missing_mask.any():
            examples = df.loc[missing_mask, audio_col].head(5).tolist()
            raise FileNotFoundError(
                f"{missing_mask.sum()} audio files missing. First examples: {examples}"
            )
    samples = [build_sample(row, modality, audio_col=audio_col) for _, row in df.iterrows()]
    print(f"\nRunning inference on {len(samples)} samples...")
    result = model.predict(samples=samples, gpus=1, num_workers=0, batch_size=batch_size)
    df = df.copy()
    df["model_score"] = result.scores
    return df


def pairwise_accuracy(df, key_col="audio_path", score_col="model_score"):
    correct = df[df["score"] == 100].set_index(key_col)
    wrong   = df[df["score"] == 0  ].set_index(key_col)
    shared  = correct.index.intersection(wrong.index)
    if len(shared) == 0:
        return float("nan"), float("nan"), 0
    wins = (correct.loc[shared, score_col].values > wrong.loc[shared, score_col].values).sum()
    gap  = (correct.loc[shared, score_col].values - wrong.loc[shared, score_col].values).mean()
    return wins / len(shared), gap, len(shared)


def load_contraprost_csv_files(data_dir: str) -> pd.DataFrame:
    """Load all en_*_expanded.csv files, tagging lang and joining category from original data."""
    pattern = os.path.join(data_dir, "en_*_expanded.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No en_*_expanded.csv files found in {data_dir}")

    # Audio paths in the CSVs are relative to the ml-speech-is-more-than-words repo root
    audio_root = os.path.abspath(os.path.join(data_dir, "ml-speech-is-more-than-words"))

    # Build category lookup using absolute paths as keys (same resolution as src_audio below)
    orig_dir = os.path.join(audio_root, "data")
    audio_to_category: dict = {}
    if os.path.isdir(orig_dir):
        for orig_path in sorted(glob.glob(os.path.join(orig_dir, "en_*.csv"))):
            orig = pd.read_csv(orig_path)
            for _, row in orig.iterrows():
                audio_to_category[os.path.join(audio_root, row["audio_1"])] = row["category"]
                audio_to_category[os.path.join(audio_root, row["audio_2"])] = row["category"]

    frames = []
    for path in csv_files:
        m = re.match(r"en_([a-z]+)_expanded\.csv", os.path.basename(path))
        lang = m.group(1) if m else os.path.basename(path)
        df = pd.read_csv(path)
        df["src_audio"] = df["src_audio"].apply(lambda p: os.path.join(audio_root, p))
        df["lang"] = lang
        df["category"] = df["src_audio"].map(audio_to_category).fillna("Unknown")
        missing = (~df["src_audio"].apply(os.path.exists)).sum()
        if missing > 0:
            raise FileNotFoundError(
                f"{missing} audio files missing for {os.path.basename(path)}. "
                f"Expected under {audio_root}"
            )
        empty = df["src_audio"].apply(lambda p: os.path.getsize(p) == 0)
        if empty.any():
            print(f"  WARNING: skipping {empty.sum()} row(s) with empty audio in {os.path.basename(path)}: "
                  + ", ".join(df.loc[empty, "src_audio"].tolist()))
            df = df[~empty].reset_index(drop=True)
        frames.append(df)
        cats = sorted(df["category"].unique().tolist())
        print(f"  Loaded {len(df):4d} rows  en_{lang}_expanded.csv  (categories={cats})")

    return pd.concat(frames, ignore_index=True)
