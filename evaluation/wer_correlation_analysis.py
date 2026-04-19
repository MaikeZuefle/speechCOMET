"""
WER-stratified correlation analysis.
For each WER bucket, computes segment-level and system-level correlation
between model QE scores and human scores, for en-de and en-zh.

Usage:
    python scripts/04-wer_correlation_analysis.py \
        --model-dir trained_models/harris-20ep-continue \
        --split dev_asr
"""
import argparse
import collections
import json
import statistics
import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import subset2evaluate.evaluate


MODEL_ORDER = [
    "lewis",
    "skye",
    "harris",
    "shetland",
    "orkney-avg",
    "orkney-sum",
    "orkney-concat",
    "orkney-sum-from-text-ckpt",
    "mull-avg",
    "mull-attn",
]

def model_sort_key(model_name):
    for i, prefix in enumerate(MODEL_ORDER):
        if model_name.startswith(prefix):
            return i
    return len(MODEL_ORDER)  # unknown models go to the end


BUCKETS = [
    ("≤0.1",   0.0,  0.1),
    ("0.1–0.3", 0.1,  0.3),
    ("0.3–0.5", 0.3,  0.5),
    ("0.5–0.7", 0.5,  0.7),
    ("0.7–1.0", 0.7,  1.0),
    (">1.0",    1.0, float("inf")),
]

LANG_PAIRS = ["en-de", "en-zh"]


def segment_level(data_lang):
    docs = collections.defaultdict(list)
    for line in data_lang:
        docs[line["doc_id"]].append(line)
    corrs = [
        scipy.stats.kendalltau(
            [line["score"] for line in doc],
            [line["score_pred"] for line in doc],
            variant="b",
        ).correlation
        for doc in docs.values()
        if len(doc) >= 2
    ]
    valid = [x for x in corrs if not np.isnan(x)]
    return statistics.mean(valid) if valid else float("nan")


def wer_vs_corr(data_lang):
    """Per-document: correlate avg WER with segment-level Kendall tau.
    Returns Spearman r (and p-value) across documents."""
    docs = collections.defaultdict(list)
    for line in data_lang:
        if line.get("wer") is not None:
            docs[line["doc_id"]].append(line)

    doc_wers, doc_corrs = [], []
    for doc in docs.values():
        if len(doc) < 2:
            continue
        tau = scipy.stats.kendalltau(
            [l["score"] for l in doc],
            [l["score_pred"] for l in doc],
            variant="b",
        ).correlation
        if np.isnan(tau):
            continue
        doc_wers.append(np.mean([l["wer"] for l in doc]))
        doc_corrs.append(tau)

    if len(doc_wers) < 3:
        return float("nan"), float("nan")
    r, p = scipy.stats.spearmanr(doc_wers, doc_corrs)
    return r, p


def system_level(data_lang):
    data_coll = collections.defaultdict(list)
    for line in data_lang:
        data_coll[line["doc_id"]].append(line)
    data_coll = [
        {"scores": {line["tgt_system"]: {"score": line["score"], "score_pred": line["score_pred"]} for line in doc}}
        for doc in data_coll.values()
    ]
    systems = set.union(*[set(doc["scores"].keys()) for doc in data_coll])
    data_coll = [doc for doc in data_coll if set(doc["scores"].keys()) == systems]
    for doc in data_coll:
        doc["scores"] = {s: doc["scores"][s] for s in systems}
    if not data_coll:
        return float("nan")
    return subset2evaluate.evaluate.eval_subset_spa(data_coll, data_coll, metric=("score", "score_pred"))


def load_data(model_dir, split, lang_pair, score_suffix=""):
    input_file  = os.path.join(model_dir, f"input_data_{split}_{lang_pair}.jsonl")
    suffix = f"_{score_suffix}" if score_suffix else ""
    scores_file = os.path.join(model_dir, f"output_scores_{split}_{lang_pair}{suffix}.jsonl")

    with open(input_file)  as f: data   = [json.loads(l) for l in f]
    with open(scores_file) as f: scores = [json.loads(l) for l in f]

    if len(data) != len(scores):
        raise ValueError(f"Length mismatch for {lang_pair}")

    for line, score in zip(data, scores):
        line["score_pred"] = score
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--split", default="dev_asr")
    parser.add_argument("--score-suffix", default="",
                        help="Suffix on score files, e.g. 'audio' for output_scores_..._audio.jsonl")
    parser.add_argument("--wer-csv", default="data/wer_dev_asr.csv")
    parser.add_argument("--results-csv", default="data/wer_analysis/wer_correlation_results.csv")
    parser.add_argument("--output", default=None, help="Plot output path. Defaults to data/wer_analysis/<model_name>.png")
    args = parser.parse_args()

    wer_df = pd.read_csv(args.wer_csv)
    wer_lookup = {(row.audio_path, row.doc_id, row.tgt_system, row.tgt_lang): row.wer for row in wer_df.itertuples()}

    results = {}  # lang_pair -> bucket_label -> {seg, sys, n}

    for lang_pair in LANG_PAIRS:
        input_file = os.path.join(args.model_dir, f"input_data_{args.split}_{lang_pair}.jsonl")
        if not os.path.exists(input_file):
            print(f"Skipping {lang_pair} (no input file found)")
            continue

        data = load_data(args.model_dir, args.split, lang_pair, args.score_suffix)

        # attach WER
        matched, missing = 0, 0
        for line in data:
            key = (line["audio_path"], line["doc_id"], line["tgt_system"], line["tgt_lang"])
            w = wer_lookup.get(key)
            if w is None:
                missing += 1
            line["wer"] = w
            matched += (w is not None)
        if missing:
            print(f"  {lang_pair}: {missing} examples missing WER")

        results[lang_pair] = {}
        print(f"\n{lang_pair}:")
        print(f"  {'Bucket':<12} {'N':>6}  {'Seg-level':>10}  {'Sys-level':>10}")
        print(f"  {'-'*46}")

        for label, lo, hi in BUCKETS:
            subset = [l for l in data if l["wer"] is not None and lo <= l["wer"] < hi]
            n = len(subset)
            if n < 5:
                seg, sys = float("nan"), float("nan")
            else:
                seg = segment_level(subset)
                sys = system_level(subset)
            results[lang_pair][label] = {"seg": seg, "sys": sys, "n": n}
            seg_str = f"{seg:.1%}" if not np.isnan(seg) else "  n/a"
            sys_str = f"{sys:.1%}" if not np.isnan(sys) else "  n/a"
            print(f"  {label:<12} {n:>6}  {seg_str:>10}  {sys_str:>10}")

        # overall
        all_data = [l for l in data if l["wer"] is not None]
        seg_all = segment_level(all_data)
        sys_all = system_level(all_data)
        wer_r, wer_p = wer_vs_corr(all_data)
        results[lang_pair]["_all"] = {"seg": seg_all, "sys": sys_all, "wer_r": wer_r, "wer_p": wer_p}
        wer_r_str = f"{wer_r:+.3f} (p={wer_p:.3f})" if not np.isnan(wer_r) else "  n/a"
        print(f"  {'ALL':<12} {len(all_data):>6}  {seg_all:.1%}  {sys_all:.1%}")
        print(f"  WER vs seg-corr (Spearman r): {wer_r_str}")

    # Build CSV row
    bucket_labels = [b[0] for b in BUCKETS]
    os.makedirs("data/wer_analysis", exist_ok=True)
    model_name = os.path.basename(args.model_dir.rstrip("/"))
    if args.score_suffix:
        model_name = f"{model_name}_{args.score_suffix}"
    plot_path = args.output or f"data/wer_analysis/{model_name}.png"
    row = {"model": model_name}

    # overall scores
    for lp, lp_key in [("en-de", "de"), ("en-zh", "zh")]:
        if lp in results:
            row[f"segment_{lp_key}"] = results[lp]["_all"]["seg"]
            row[f"system_{lp_key}"]  = results[lp]["_all"]["sys"]
            row[f"wer_r_{lp_key}"]   = results[lp]["_all"]["wer_r"]
        else:
            row[f"segment_{lp_key}"] = float("nan")
            row[f"system_{lp_key}"]  = float("nan")
            row[f"wer_r_{lp_key}"]   = float("nan")
    segs = [row[k] for k in ["segment_de", "segment_zh"] if not np.isnan(row[k])]
    syss = [row[k] for k in ["system_de",  "system_zh"]  if not np.isnan(row[k])]
    wer_rs = [row[k] for k in ["wer_r_de", "wer_r_zh"] if not np.isnan(row[k])]
    row["segment_avg"] = statistics.mean(segs) if segs else float("nan")
    row["system_avg"]  = statistics.mean(syss) if syss else float("nan")
    row["wer_r_avg"]   = statistics.mean(wer_rs) if wer_rs else float("nan")

    # per-bucket scores
    for metric, key in [("seg", "segment"), ("sys", "system")]:
        for lp, lp_key in [("en-de", "de"), ("en-zh", "zh")]:
            for label in bucket_labels:
                col = f"{key}_{lp_key}_{label}"
                row[col] = results[lp][label][metric] if lp in results else float("nan")

    # upsert into CSV
    csv_path = args.results_csv
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if model_name in df["model"].values:
            df.loc[df["model"] == model_name, list(row.keys())[1:]] = list(row.values())[1:]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    # define explicit column order
    bucket_labels = [b[0] for b in BUCKETS]
    ordered_cols = (
        ["model"]
        + ["segment_de", "segment_zh", "segment_avg"]
        + ["system_de",  "system_zh",  "system_avg"]
        + ["wer_r_de", "wer_r_zh", "wer_r_avg"]
        + [f"segment_de_{l}" for l in bucket_labels]
        + [f"segment_zh_{l}" for l in bucket_labels]
        + [f"system_de_{l}"  for l in bucket_labels]
        + [f"system_zh_{l}"  for l in bucket_labels]
    )
    # keep only columns that exist (handles partially filled rows)
    ordered_cols = [c for c in ordered_cols if c in df.columns]

    # sort by predefined model order and round numeric columns to 4 digits
    df["_order"] = df["model"].apply(model_sort_key)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    numeric_cols = [c for c in ordered_cols if c != "model"]
    df[numeric_cols] = df[numeric_cols].round(4)
    df[ordered_cols].to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Plot
    n_langs = len(results)
    if n_langs == 0:
        print("No results to plot.")
        return

    fig, axes = plt.subplots(2, n_langs, figsize=(6 * n_langs, 8), squeeze=False)
    labels = [b[0] for b in BUCKETS]

    for col, (lang_pair, buckets) in enumerate(results.items()):
        for row, metric in enumerate(["seg", "sys"]):
            ax = axes[row][col]
            vals = [buckets[l][metric] for l in labels]
            ns   = [buckets[l]["n"]    for l in labels]
            colors = ["#4c72b0" if not np.isnan(v) else "#cccccc" for v in vals]
            bars = ax.bar(labels, [v if not np.isnan(v) else 0 for v in vals], color=colors)
            for bar, n in zip(bars, ns):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f"n={n}", ha="center", va="bottom", fontsize=7)
            ax.set_title(f"{lang_pair} — {'Segment' if metric == 'seg' else 'System'}-level")
            ax.set_ylabel("Kendall τ" if metric == "seg" else "SPA")
            ax.set_xlabel("WER bucket")
            ax.set_ylim(0, max((v for v in vals if not np.isnan(v)), default=0.1) * 1.2)
            ax.tick_params(axis="x", rotation=20)

    fig.suptitle(f"Correlation by WER bucket — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
