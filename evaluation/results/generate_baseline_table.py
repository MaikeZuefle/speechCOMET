#!/usr/bin/env python3
"""Generate colored LaTeX table for SpeechCOMET baseline results (v2 format)."""
import os

N = None

# (type_label, model,
#  seg_de, seg_zh, seg_avg, sys_de, sys_zh, sys_avg,  # human/only transcript
#  d_seg, d_sys,                                        # ASR delta (None if n/a)
#  mshe_es, mshe_fr, mshe_it,
#  cp_de, cp_es, cp_ja)

data = [
    (r"\multirow{2}{*}{\textsc{Text}}",
     r"\textsc{COMET-Partial}",
     11.20, 12.70, 11.95,  43.10, 67.00, 55.05,
     -0.35, +4.73,
     0.524, 0.557, 0.518,  0.500, 0.500, 0.500),
    ("",
     r"\textsc{COMETKiwi}",
     32.80, 36.40, 34.60,  86.30, 89.10, 87.70,
     -6.01, -0.73,
     0.615, 0.606, 0.552,  0.500, 0.500, 0.500),
    (r"\multirow{2}{*}{\textsc{Speech}}",
     r"\textsc{SpeechQE}",
     26.87, 32.73, 29.80,  79.20, 71.30, 75.25,
     N, N,
     0.370, 0.360, 0.310,  0.261, 0.200, 0.382),
    ("",
     r"\textsc{BLASER}",
     22.03, 26.81, 24.42,  85.52, 67.40, 76.46,
     N, N,
     0.520, 0.516, 0.517,  0.510, 0.515, 0.511),
]

N_SCORE = 12

COLORS = {
    "teal":   (136, 204, 203),
    "orange": (247, 136,  50),
    "pink":   (217, 110, 173),
}
INTENSITY = 0.60

def color_for(norm, scheme):
    tr, tg, tb = COLORS[scheme]
    return (int(255 + INTENSITY * (tr - 255) * norm),
            int(255 + INTENSITY * (tg - 255) * norm),
            int(255 + INTENSITY * (tb - 255) * norm))

col_vals = [[] for _ in range(N_SCORE)]
for row in data:
    vals = list(row[2:8]) + list(row[10:16])
    for i, v in enumerate(vals):
        if v is not None:
            col_vals[i].append(v)

col_min  = [min(v) for v in col_vals]
col_max  = [max(v) for v in col_vals]
col_best = [max(v) for v in col_vals]

all_deltas = [v for row in data for v in [row[8], row[9]] if v is not None]
delta_min = min(all_deltas)
delta_max = max(all_deltas)


def score_cell(v, col_idx):
    if v is None:
        return r"\phantom{00.0}--"
    mn, mx = col_min[col_idx], col_max[col_idx]
    if col_idx < 6:
        scheme, display = "teal",   f"{v:.1f}"
        norm = (v - mn) / (mx - mn) if mx > mn else 0.5
    elif col_idx < 9:
        scheme, display = "orange", f"{v * 100:.1f}"
        norm = max(0.0, (v - 0.50) / (mx - 0.50)) if mx > 0.50 else 0.0
    else:
        scheme, display = "pink",   f"{v * 100:.1f}"
        norm = max(0.0, (v - 0.50) / (mx - 0.50)) if mx > 0.50 else 0.0
    r, g, b = color_for(norm, scheme)
    if v == col_best[col_idx]:
        display = rf"\textbf{{{display}}}"
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}{display}"


def delta_cell(d):
    if d is None:
        return "--"
    sign = "+" if d >= 0 else ""
    norm = (d - delta_min) / (delta_max - delta_min) if delta_max > delta_min else 0.5
    r = int(255 + 0.5 * (58  - 255) * norm)
    g = int(255 + 0.5 * (126 - 255) * norm)
    b = int(255 + 0.5 * (192 - 255) * norm)
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}\scriptsize{{{sign}{d:.1f}}}"


lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{ll|rrrc|rrrc|rrr|rrr}",
    r"\toprule",
    (r" & & \multicolumn{8}{c|}{\textbf{IWSLT dev}}"
     r" & \multicolumn{3}{c|}{\textbf{MuST-SHE}}"
     r" & \multicolumn{3}{c}{\textbf{ContraProST}} \\"),
    (r" & & \multicolumn{4}{c|}{Segment $\tau_b$ (\%)}"
     r" & \multicolumn{4}{c|}{System SPA (\%)}"
     r" & \multicolumn{3}{c|}{PA (\%)} & \multicolumn{3}{c}{PA (\%)} \\"),
    r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-13}\cmidrule(lr){14-16}",
    (r" & & \textbf{de} & \textbf{zh} & \textbf{avg} & \multicolumn{1}{c|}{\textbf{ASR src}}"
     r" & \textbf{de} & \textbf{zh} & \textbf{avg} & \multicolumn{1}{c|}{\textbf{ASR src}}"
     r" & \textbf{es} & \textbf{fr} & \textbf{it}"
     r" & \textbf{de} & \textbf{es} & \textbf{ja} \\"),
    r"\midrule",
]

for row in data:
    type_col, model = row[0], row[1]
    seg_de, seg_zh, seg_avg, sys_de, sys_zh, sys_avg = row[2:8]
    d_seg, d_sys = row[8], row[9]
    mshe = row[10:13]
    cp   = row[13:16]

    s_cols = [seg_de, seg_zh, seg_avg, sys_de, sys_zh, sys_avg] + list(mshe) + list(cp)
    cells = [type_col, model]
    for i, v in enumerate(s_cols[:6]):
        cells.append(score_cell(v, i))
        if i == 2:
            cells.append(delta_cell(d_seg))
        if i == 5:
            cells.append(delta_cell(d_sys))
    for i, v in enumerate(mshe):
        cells.append(score_cell(v, 6 + i))
    for i, v in enumerate(cp):
        cells.append(score_cell(v, 9 + i))
    lines.append(" & ".join(cells) + r" \\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    (r"\caption{Results for transcript- and speech-based baseline metrics."
     r" Transcript-based metrics cannot evaluate speech-specific phenomena by design;"
     r" speech-based metrics can access the signal but still fail on gender and prosody.}"
     r"\label{tab:baselines}"),
    r"\end{table*}",
]

output = "\n".join(lines)
out_path = os.path.join(os.path.dirname(__file__), "table_baselines.tex")
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"Saved to {out_path}")
