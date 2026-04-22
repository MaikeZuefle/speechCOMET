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


def load_data(model_dir, split, lang_pair):
    input_file  = os.path.join(model_dir, f"input_data_{split}_{lang_pair}.jsonl")
    scores_file = os.path.join(model_dir, f"output_scores_{split}_{lang_pair}.jsonl")

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
    parser.add_argument("--wer-csv", default=None)
    parser.add_argument("--results-csv", default="data/wer_correlation_results.csv")
    parser.add_argument("--output", default="data/wer_correlation.png")
    args = parser.parse_args()

    wer_df = pd.read_csv(args.wer_csv)
    wer_lookup = {(row.audio_path, row.doc_id): row.wer for row in wer_df.itertuples()}

    results = {}  # lang_pair -> bucket_label -> {seg, sys, n}

    for lang_pair in LANG_PAIRS:
        input_file = os.path.join(args.model_dir, f"input_data_{args.split}_{lang_pair}.jsonl")
        if not os.path.exists(input_file):
            print(f"Skipping {lang_pair} (no input file found)")
            continue

        data = load_data(args.model_dir, args.split, lang_pair)

        # attach WER
        matched, missing = 0, 0
        for line in data:
            key = (line["audio_path"], line["doc_id"])
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
        results[lang_pair]["_all"] = {"seg": seg_all, "sys": sys_all}
        print(f"  {'ALL':<12} {len(all_data):>6}  {seg_all:.1%}  {sys_all:.1%}")

    # Build CSV row
    bucket_labels = [b[0] for b in BUCKETS]
    model_name = os.path.basename(args.model_dir.rstrip("/"))
    row = {"model": model_name}

    # overall scores
    for lp, lp_key in [("en-de", "de"), ("en-zh", "zh")]:
        if lp in results:
            row[f"segment_{lp_key}"] = results[lp]["_all"]["seg"]
            row[f"system_{lp_key}"]  = results[lp]["_all"]["sys"]
        else:
            row[f"segment_{lp_key}"] = float("nan")
            row[f"system_{lp_key}"]  = float("nan")
    segs = [row[k] for k in ["segment_de", "segment_zh"] if not np.isnan(row[k])]
    syss = [row[k] for k in ["system_de",  "system_zh"]  if not np.isnan(row[k])]
    row["segment_avg"] = statistics.mean(segs) if segs else float("nan")
    row["system_avg"]  = statistics.mean(syss) if syss else float("nan")

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
    df.to_csv(csv_path, index=False)
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
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
