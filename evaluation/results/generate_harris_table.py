#!/usr/bin/env python3
"""Ablation table: Harris architecture variants (additive / COMET-style / joint encoding).

Model mapping:
  harris-20ep   → speech_concat_metric, fuse=sum   → additive
  concat-harris → speech_regression_metric          → COMET-style
  joint-harris  → speech_joint_metric               → joint encoding
All models: SONAR audio input, IWSLT fine-tuned, dev_asr split.
"""
import os

N = None

# (arch_label,
#  seg_de, seg_zh, seg_avg, sys_de, sys_zh, sys_avg,
#  mshe_es, mshe_fr, mshe_it, cp_de, cp_es, cp_ja)

data = [
    ("four-way",
     14.6, 22.5, 18.55,  37.2, 80.4, 58.80,
     0.560, 0.519, 0.513,  0.500, 0.500, 0.500),
    ("additive",
     13.5, 22.3, 17.90,  35.3, 67.4, 51.35,
     0.546, 0.497, 0.522,  0.500, 0.500, 0.500),
    ("joint encoding",
     3.9,  6.9,  5.40,   19.9, 88.6, 54.25,
     0.516, 0.503, 0.528,  0.506, 0.492, 0.500),
]

N_SCORE = 12
COLORS = {"teal": (136, 204, 203), "orange": (247, 136, 50), "pink": (217, 110, 173)}
INTENSITY = 0.60


def color_for(norm, scheme):
    tr, tg, tb = COLORS[scheme]
    return (int(255 + INTENSITY * (tr - 255) * norm),
            int(255 + INTENSITY * (tg - 255) * norm),
            int(255 + INTENSITY * (tb - 255) * norm))


col_vals = [[] for _ in range(N_SCORE)]
for row in data:
    for i, v in enumerate(list(row[1:7]) + list(row[7:13])):
        if v is not None:
            col_vals[i].append(v)

col_min  = [min(v) for v in col_vals]
col_max  = [max(v) for v in col_vals]
col_best = [max(v) for v in col_vals]


def cell(v, col_idx):
    if v is None:
        return r"\phantom{00.0}--"
    mn, mx = col_min[col_idx], col_max[col_idx]
    if col_idx < 6:
        norm = (v - mn) / (mx - mn) if mx > mn else 0.5
        scheme, display = "teal",   f"{v:.1f}"
    elif col_idx < 9:
        norm = max(0.0, (v - 0.50) / (mx - 0.50)) if mx > 0.50 else 0.0
        scheme, display = "orange", f"{v * 100:.1f}"
    else:
        norm = max(0.0, (v - 0.50) / (mx - 0.50)) if mx > 0.50 else 0.0
        scheme, display = "pink",   f"{v * 100:.1f}"
    r, g, b = color_for(norm, scheme)
    if v == col_best[col_idx]:
        display = rf"\textbf{{{display}}}"
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}{display}"


lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{l|rrr|rrr|rrr|rrr}",
    r"\toprule",
    (r" & \multicolumn{6}{c|}{\textbf{IWSLT dev}}"
     r" & \multicolumn{3}{c|}{\textbf{MuST-SHE}}"
     r" & \multicolumn{3}{c}{\textbf{ContraProST}} \\"),
    (r" & \multicolumn{3}{c|}{Segment $\tau_b$ (\%)}"
     r" & \multicolumn{3}{c|}{System SPA (\%)}"
     r" & \multicolumn{3}{c|}{PA (\%)}"
     r" & \multicolumn{3}{c}{PA (\%)} \\"),
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
    (r"\textbf{Architecture}"
     r" & \textbf{de} & \textbf{zh} & \textbf{avg}"
     r" & \textbf{de} & \textbf{zh} & \textbf{avg}"
     r" & \textbf{es} & \textbf{fr} & \textbf{it}"
     r" & \textbf{de} & \textbf{es} & \textbf{ja} \\"),
    r"\midrule",
]

for k, row in enumerate(data):
    arch = row[0]
    vals = list(row[1:7]) + list(row[7:13])
    cells = [arch]
    for i, v in enumerate(vals):
        cells.append(cell(v, i))
    lines.append(" & ".join(cells) + r" \\")
    if k < len(data) - 1:
        lines.append(r"\midrule")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    (r"\caption{Ablation of audio--text integration strategies for speech QE."
     r" \emph{Four-way}: standard COMET estimator input"
     r" $[\mathbf{h}_t; \mathbf{s}_a; \mathbf{h}_t \odot \mathbf{s}_a; |\mathbf{h}_t - \mathbf{s}_a|]$,"
     r" with SONAR audio replacing the source text embedding (used in the main model)."
     r" \emph{Additive}: SONAR audio embedding summed with the unified CLS MT embedding,"
     r" following the COMETKiwi unified-metric architecture."
     r" \emph{Joint encoding}: SONAR audio embedding injected as a virtual token into XLM-R self-attention."
     r" All models are speech-only (SONAR audio input) fine-tuned on IWSLT.}"
     r"\label{tab:harris_ablation}"),
    r"\end{table*}",
]

output = "\n".join(lines)
out_path = os.path.join(os.path.dirname(__file__), "table_harris_ablation.tex")
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"Saved to {out_path}")
