#!/usr/bin/env python3
"""Ablation table: Orkney (SONAR speech+text) fusion strategy and encoder init."""
import os

# (fusion_label, [(variant_label, seg_de, seg_zh, seg_avg, sys_de, sys_zh, sys_avg,
#                  mshe_es, mshe_fr, mshe_it, cp_de, cp_es, cp_ja), ...])

SEP = "---"

groups = [
    ("avg", [
        ("frozen",
         16.99, 24.53, 20.76,  64.50, 67.70, 66.10,
         0.528, 0.530, 0.507,  0.501, 0.501, 0.500),
    ]),
    SEP,
    ("concat", [
        ("frozen",
         12.20, 21.07, 16.64,  48.98, 75.70, 62.34,
         0.524, 0.486, 0.513,  0.500, 0.500, 0.500),
    ]),
    SEP,
    ("sum", [
        ("frozen",
         20.87, 24.70, 22.79,  70.17, 67.40, 68.79,
         0.548, 0.554, 0.530,  0.500, 0.500, 0.500),
        (r"\quad WMT text pretrain $\to$ FT (frozen)",
         24.5, 27.1, 25.80,  79.7, 67.4, 73.55,
         0.556, 0.526, 0.524,  0.500, 0.499, 0.500),
        (r"\quad WMT text pretrain $\to$ FT (unfrozen)",
         23.4, 27.1, 25.25,  80.2, 67.4, 73.80,
         0.560, 0.544, 0.530,  0.500, 0.500, 0.500),
    ]),
]

N_SCORE = 12
COLORS = {"teal": (136, 204, 203), "orange": (247, 136, 50), "pink": (217, 110, 173)}
INTENSITY = 0.60


def color_for(norm, scheme):
    tr, tg, tb = COLORS[scheme]
    return (int(255 + INTENSITY * (tr - 255) * norm),
            int(255 + INTENSITY * (tg - 255) * norm),
            int(255 + INTENSITY * (tb - 255) * norm))


all_rows = [v for g in groups if g is not SEP for v in g[1]]
col_vals = [[] for _ in range(N_SCORE)]
for row in all_rows:
    for i, v in enumerate(list(row[1:7]) + list(row[7:13])):
        col_vals[i].append(v)

col_min  = [min(v) for v in col_vals]
col_max  = [max(v) for v in col_vals]
col_best = [max(v) for v in col_vals]


def cell(v, col_idx):
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
    r"\begin{tabular}{ll|rrr|rrr|rrr|rrr}",
    r"\toprule",
    (r" & & \multicolumn{6}{c|}{\textbf{IWSLT dev}}"
     r" & \multicolumn{3}{c|}{\textbf{MuST-SHE}}"
     r" & \multicolumn{3}{c}{\textbf{ContraProST}} \\"),
    (r" & & \multicolumn{3}{c|}{Segment $\tau_b$ (\%)}"
     r" & \multicolumn{3}{c|}{System SPA (\%)}"
     r" & \multicolumn{3}{c|}{PA (\%)}"
     r" & \multicolumn{3}{c}{PA (\%)} \\"),
    r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}",
    (r"\textbf{Fusion} & \textbf{SONAR}"
     r" & \textbf{de} & \textbf{zh} & \textbf{avg}"
     r" & \textbf{de} & \textbf{zh} & \textbf{avg}"
     r" & \textbf{es} & \textbf{fr} & \textbf{it}"
     r" & \textbf{de} & \textbf{es} & \textbf{ja} \\"),
    r"\midrule",
]

for g in groups:
    if g is SEP:
        lines.append(r"\cmidrule(l){1-14}")
        continue
    fusion_label, variants = g
    n = len(variants)
    for k, row in enumerate(variants):
        variant_label = row[0]
        vals = list(row[1:7]) + list(row[7:13])
        if k == 0:
            fusion_cell = (rf"\multirow{{{n}}}{{*}}{{{fusion_label}}}"
                           if n > 1 else fusion_label)
        else:
            fusion_cell = ""
        cells = [fusion_cell, variant_label]
        for i, v in enumerate(vals):
            cells.append(cell(v, i))
        lines.append(" & ".join(cells) + r" \\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    (r"\caption{Ablation of embedding fusion strategies for SpeechCOMET with SONAR encoder."
     r" \emph{Fusion} controls how speech and text embeddings are combined;"
     r" \emph{SONAR} indicates whether the encoder is frozen or unfrozen during fine-tuning."
     r" Text init initialises from the WMT-pretrained text SpeechCOMET checkpoint.}"
     r"\label{tab:orkney_ablation}"),
    r"\end{table*}",
]

output = "\n".join(lines)
out_path = os.path.join(os.path.dirname(__file__), "table_orkney_ablation.tex")
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"Saved to {out_path}")
