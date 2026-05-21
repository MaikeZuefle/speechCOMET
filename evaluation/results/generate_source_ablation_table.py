#!/usr/bin/env python3
"""Appendix table: source-modality ablation for speech+text SpeechCOMET models.

Reads correlation files from each model's shuffled_src / shuffled_audio /
shuffled_text subdirectories and shows how segment τ_b changes when each
source modality is individually replaced with a mismatched one.

Run after: bash evaluation/scripts/shuffled_src/run_shuffled_src.sh
"""
import os
import re

TRAINED_MODELS = "trained_models"
LANGS = ["en-de", "en-zh"]

# (paper_name, folder, split)
MODELS = [
    (r"SpeechCOMET",          "orkney-sum-from-text-ckpt-20ep",    "dev"),
    (r"SpeechCOMET$^\dagger$", "orkney-sum-from-text-ckpt-BIG",     "dev"),
]

SUBDIRS = {
    "real":  None,           # root correlation files
    "both":  "shuffled_src",
    "audio": "shuffled_audio",
    "text":  "shuffled_text",
}


def read_seg_avg(model_folder, subdir, split):
    """Return (seg_avg, sys_avg) averaged over LANGS, or (None, None) if missing."""
    seg_vals, sys_vals = [], []
    for lang in LANGS:
        if subdir is None:
            corr_file = os.path.join(model_folder, f"correlation_{split}_{lang}.txt")
        else:
            corr_file = os.path.join(model_folder, subdir, f"correlation_{split}_{lang}.txt")
        if not os.path.exists(corr_file):
            return None, None
        text = open(corr_file).read()
        segs = re.findall(r"SEGMENT-LEVEL.*?(\d+\.\d+)%", text, re.S)
        syss = re.findall(r"SYSTEM-LEVEL.*?(\d+\.\d+)%", text, re.S)
        if not segs or not syss:
            return None, None
        seg_vals.append(float(segs[0]))
        sys_vals.append(float(syss[0]))
    if len(seg_vals) < len(LANGS):
        return None, None
    return sum(seg_vals) / len(seg_vals), sum(sys_vals) / len(sys_vals)


# ── colour helpers ─────────────────────────────────────────────────────────────
def delta_cell(d):
    if d is None:
        return r"\cellcolor[RGB]{220,230,245}--"
    sign = "+" if d >= 0 else ""
    intensity = min(abs(d) / 40.0, 1.0) * 0.55
    if d < 0:
        r = int(255 + intensity * (58  - 255))
        g = int(255 + intensity * (126 - 255))
        b = int(255 + intensity * (192 - 255))
    else:
        r = int(255 + intensity * (220 - 255))
        g = int(255 + intensity * (160 - 255))
        b = int(255 + intensity * (130 - 255))
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}{sign}{d:.1f}"


def score_cell(v, mn, mx):
    if v is None:
        return r"\cellcolor[RGB]{220,230,245}--"
    norm = (v - mn) / (mx - mn) if mx > mn else 0.5
    r = int(255 + 0.60 * (136 - 255) * norm)
    g = int(255 + 0.60 * (204 - 255) * norm)
    b = int(255 + 0.60 * (203 - 255) * norm)
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}{v:.1f}"


# ── collect data ──────────────────────────────────────────────────────────────
rows = []
for paper_name, folder, split in MODELS:
    model_folder = os.path.join(TRAINED_MODELS, folder)
    scores = {}
    for key, subdir in SUBDIRS.items():
        seg, sys = read_seg_avg(model_folder, subdir, split)
        scores[key] = (seg, sys)
    rows.append((paper_name, scores))

# colour scale over all real seg scores that exist
real_segs = [r[1]["real"][0] for r in rows if r[1]["real"][0] is not None]
mn = min(real_segs) if real_segs else 0
mx = max(real_segs) if real_segs else 1

# ── build table ───────────────────────────────────────────────────────────────
lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{l|rrrr}",
    r"\toprule",
    (r"\textbf{Model}"
     r" & \textbf{real src}"
     r" & \textbf{$\Delta$ both}"
     r" & \textbf{$\Delta$ audio only}"
     r" & \textbf{$\Delta$ text only} \\"),
    r"\cmidrule(lr){2-2}\cmidrule(lr){3-5}",
    r"\multicolumn{5}{l}{\textit{Segment $\tau_b$ avg (\%) over de \& zh}} \\",
    r"\midrule",
]

for paper_name, scores in rows:
    seg_real = scores["real"][0]
    d_both  = (scores["both"][0]  - seg_real) if scores["both"][0]  is not None and seg_real is not None else None
    d_audio = (scores["audio"][0] - seg_real) if scores["audio"][0] is not None and seg_real is not None else None
    d_text  = (scores["text"][0]  - seg_real) if scores["text"][0]  is not None and seg_real is not None else None

    cells = [
        paper_name,
        score_cell(seg_real, mn, mx),
        delta_cell(d_both),
        delta_cell(d_audio),
        delta_cell(d_text),
    ]
    lines.append(" & ".join(cells) + r" \\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    (r"\caption{Source-modality ablation for speech+text SpeechCOMET models."
     r" $\Delta$ audio only: audio replaced with mismatched audio, text kept real."
     r" $\Delta$ text only: text replaced with mismatched text, audio kept real."
     r" $\Delta$ both: both modalities replaced."
     r" Large negative $\Delta$ indicates the model uses that modality."
     r" Scores are segment $\tau_b$ averaged over de and zh.}"
     r"\label{tab:source_ablation}"),
    r"\end{table}",
]

output = "\n".join(lines)
out_path = os.path.join(os.path.dirname(__file__), "table_source_ablation.tex")
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"Saved to {out_path}")

# Print a quick summary to the console
print("\nCurrent data status:")
for paper_name, scores in rows:
    print(f"\n  {paper_name}:")
    for key in ["real", "both", "audio", "text"]:
        seg, sys = scores[key]
        status = f"seg={seg:.1f}  sys={sys:.1f}" if seg is not None else "MISSING — run ablation first"
        print(f"    {key:6s}: {status}")
