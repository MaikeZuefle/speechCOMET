#!/usr/bin/env python3
"""Ablation table: SpeechLLM standard vs speech-aware prompt on MuST-SHE and ContraProST."""
import os

# (type, model_label,
#  mshe_es, mshe_fr, mshe_it,
#  cp_de, cp_es, cp_ja)

data = [
    # ── Speech ───────────────────────────────────────────────────────────────
    (r"\multirow{3}{*}{\textsc{Speech}}",
     r"SpeechLLM",
     0.319, 0.283, 0.322,  0.316, 0.284, 0.316),
    ("",
     r"\quad speech prompt",
     0.380, 0.323, 0.324,  0.153, 0.194, 0.168),
    ("",
     r"\quad+FT",
     0.359, 0.150, 0.207,  0.208, 0.221, 0.153),
    # ── Text ─────────────────────────────────────────────────────────────────
    (r"\multirow{3}{*}{\textsc{Text}}",
     r"SpeechLLM",
     0.294, 0.259, 0.236,  0.238, 0.186, 0.296),
    ("",
     r"\quad speech prompt",
     0.237, 0.173, 0.162,  0.189, 0.185, 0.157),
    ("",
     r"\quad+FT",
     0.140, 0.138, 0.154,  0.141, 0.141, 0.250),
    # ── Speech+Text ───────────────────────────────────────────────────────────
    (r"\multirow{3}{*}{\textsc{Sp.+Txt}}",
     r"SpeechLLM",
     0.380, 0.309, 0.345,  0.305, 0.264, 0.313),
    ("",
     r"\quad speech prompt",
     0.431, 0.371, 0.378,  0.228, 0.268, 0.214),
    ("",
     r"\quad+FT",
     0.082, 0.063, 0.068,  0.180, 0.093, 0.197),
]

N_SCORE = 6

COLORS = {
    "orange": (247, 136,  50),   # MuST-SHE
    "pink":   (217, 110, 173),   # ContraProST
}
INTENSITY = 0.60

def color_for(norm, scheme):
    tr, tg, tb = COLORS[scheme]
    return (int(255 + INTENSITY * (tr - 255) * norm),
            int(255 + INTENSITY * (tg - 255) * norm),
            int(255 + INTENSITY * (tb - 255) * norm))

col_vals = [[] for _ in range(N_SCORE)]
for row in data:
    for i, v in enumerate(row[2:]):
        col_vals[i].append(v)

col_min  = [min(v) for v in col_vals]
col_max  = [max(v) for v in col_vals]
col_best = [max(v) for v in col_vals]


def cell(v, col_idx):
    mn, mx = col_min[col_idx], col_max[col_idx]
    norm = max(0.0, (v - 0.50) / (mx - 0.50)) if mx > 0.50 else 0.0
    scheme = "orange" if col_idx < 3 else "pink"
    display = f"{v * 100:.1f}"
    r, g, b = color_for(norm, scheme)
    if v == col_best[col_idx]:
        display = rf"\textbf{{{display}}}"
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}{display}"


lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{ll|rrr|rrr}",
    r"\toprule",
    (r" & & \multicolumn{3}{c|}{\textbf{MuST-SHE PA (\%)}}"
     r" & \multicolumn{3}{c}{\textbf{ContraProST PA (\%)}} \\"),
    r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
    (r"\textbf{Type} & \textbf{Model}"
     r" & \textbf{es} & \textbf{fr} & \textbf{it}"
     r" & \textbf{de} & \textbf{es} & \textbf{ja} \\"),
    r"\midrule",
]

prev_type = None
for row in data:
    type_col, model = row[0], row[1]
    vals = row[2:]
    if type_col != "" and prev_type is not None:
        lines.append(r"\midrule")
    if type_col != "":
        prev_type = type_col
    cells = [type_col, model]
    for i, v in enumerate(vals):
        cells.append(cell(v, i))
    lines.append(" & ".join(cells) + r" \\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    (r"\caption{Effect of speech-aware prompting on SpeechLLM across input modalities."
     r" The speech prompt instructs the model to consider paralinguistic cues;"
     r" +FT denotes fine-tuning on IWSLT.}"
     r"\label{tab:speechllm_prompt}"),
    r"\end{table*}",
]

output = "\n".join(lines)
out_path = os.path.join(os.path.dirname(__file__), "table_speechllm_prompts.tex")
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"Saved to {out_path}")
