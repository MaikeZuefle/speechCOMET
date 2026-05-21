#!/usr/bin/env python3
"""Main results table v2: one row per model, ASR-src avg columns instead of separate ASR rows."""
import os

N = None

# Data: (section, type, model,
#        seg_de, seg_zh, seg_avg, sys_de, sys_zh, sys_avg,   # human/only transcript
#        asr_seg_avg, asr_sys_avg,                            # ASR avg scores (None if not applicable)
#        mshe_es, mshe_fr, mshe_it,
#        cp_de, cp_es, cp_ja)

BASELINE = "Baseline"
SCOMET = "SpeechCOMET"
SLLM   = "SpeechLLM"
SEP    = "---"

data = [
    # ── Baseline ─────────────────────────────────────────────────────────────
    (BASELINE, r"\textsc{Text}",
     r"\textsc{COMETKiwi}",
     32.80, 36.40, 34.60,  86.30, 89.10, 87.70,
     -6.01, -0.73,
     0.615, 0.606, 0.552,  0.500, 0.500, 0.500),
    # ── SpeechCOMET ──────────────────────────────────────────────────────────
    (SCOMET, r"\multirow{2}{*}{\textsc{Text}}",
     r"COMETKiwi$_{\text{RoBERTa}}^{\text{WMT}}$",
     20.30, 25.89, 23.10,  44.67, 67.40, 56.04,
     -1.54, +1.47,
     0.512, 0.528, 0.501,  0.500, 0.500, 0.500),
    (SCOMET, "",
     r"COMETKiwi$_{\text{RoBERTa}}^{\text{IWSLT}}$",
     20.21, 25.28, 22.75,  69.93, 67.40, 68.67,
     -0.47, -1.83,
     0.528, 0.491, 0.520,  0.500, 0.500, 0.500),
    (SCOMET, r"\multirow{2}{*}{\textsc{Speech}}",
     r"SpeechCOMET\textsubscript{SONAR}",
     17.34, 22.86, 20.10,  51.05, 70.90, 60.98,
     N, N,
     0.531, 0.509, 0.509,  0.500, 0.500, 0.500),
    (SCOMET, "",
     r"SpeechCOMET\textsubscript{Whisper}",
     16.83, 18.72, 17.78,  44.92, 68.50, 56.71,
     N, N,
     0.514, 0.507, 0.528,  0.500, 0.500, 0.501),
    (SCOMET, r"\textsc{Sp.+Txt}",
     r"SpeechCOMET",
     24.5, 27.1, 25.80,  79.7, 67.4, 73.55,
     -3.75, -1.40,
     0.556, 0.526, 0.524,  0.500, 0.499, 0.500),
    SEP,
    (SCOMET, r"\textsc{Sp.+Txt}",
     r"SpeechCOMET$^\dagger$",
     32.7, 36.0, 34.35,  85.2, 67.8, 76.5,
     -5.95, +6.55,
     0.546, 0.580, 0.511,  0.500, 0.500, 0.500),
    # ── SpeechLLM ────────────────────────────────────────────────────────────
    (SLLM, r"\multirow{2}{*}{\textsc{Text}}",
     r"SpeechLLM",
     33.20, 47.50, 40.35,  95.60, 32.60, 64.10,
     -5.55, +2.10,
     0.237, 0.173, 0.162,  0.238, 0.181, 0.311),
    (SLLM, "",
     r"\quad+FT",
     39.30, 55.00, 47.15,  93.40, 83.70, 88.55,
     -3.79, -3.36,
     0.140, 0.138, 0.154,  0.140, 0.140, 0.261),
    (SLLM, r"\multirow{2}{*}{\textsc{Speech}}",
     r"SpeechLLM",
     26.50, 37.50, 32.00,  87.50, 32.70, 60.10,
     N, N,
     0.380, 0.323, 0.324,  0.322, 0.284, 0.331),
    (SLLM, "",
     r"\quad+FT",
     33.77, 46.46, 40.12,  90.17, 50.10, 70.14,
     N, N,
     0.359, 0.150, 0.207,  0.211, 0.219, 0.157),
    (SLLM, r"\multirow{2}{*}{\textsc{Sp.+Txt}}",
     r"SpeechLLM",
     32.20, 44.10, 38.15,  90.70, 32.60, 61.65,
     -4.55, -0.55,
     0.431, 0.371, 0.378,  0.310, 0.261, 0.326),
    (SLLM, "",
     r"\quad+FT",
     39.70, 60.10, 49.90,  89.10, 72.10, 80.60,
     -4.64, -7.17,
     0.082, 0.063, 0.068,  0.178, 0.088, 0.204),
]

# ── colour scheme ─────────────────────────────────────────────────────────────
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

# ── per-column min/max/best ───────────────────────────────────────────────────
N_SCORE = 12  # seg_de..cp_ja (excluding delta cols, 6 IWSLT + 3 mustshe + 3 cp)
col_vals = [[] for _ in range(N_SCORE)]
col_vals_per_sec = {BASELINE: [[] for _ in range(N_SCORE)],
                    SCOMET:   [[] for _ in range(N_SCORE)],
                    SLLM:     [[] for _ in range(N_SCORE)]}
for row in data:
    if row is SEP:
        continue
    vals = list(row[3:9]) + list(row[11:17])  # skip delta cols (idx 9,10)
    for i, v in enumerate(vals):
        if v is not None:
            col_vals[i].append(v)
            col_vals_per_sec[row[0]][i].append(v)

col_min  = [min(v) for v in col_vals]
col_max  = [max(v) for v in col_vals]
col_best_per_sec = {
    sec: [max(v) if v else None for v in vlist]
    for sec, vlist in col_vals_per_sec.items()
}

# delta range for ASR src coloring (b): exclude None, min→white, max→darkest
all_deltas = [row[9]  for row in data if row is not SEP and row[9]  is not None] + \
             [row[10] for row in data if row is not SEP and row[10] is not None]
delta_min = min(all_deltas)
delta_max = max(all_deltas)


def score_cell(v, col_idx, section):
    if v is None:
        return r"\phantom{00.0}--"
    mn, mx = col_min[col_idx], col_max[col_idx]
    if col_idx < 6:
        scheme, display = "teal",   f"{v:.1f}"
        norm = (v - mn) / (mx - mn) if mx > mn else 0.5
    elif col_idx < 9:  # MuST-SHE: floor at 50% chance level
        scheme, display = "orange", f"{v * 100:.1f}"
        norm = max(0.0, (v - 0.50) / (mx - 0.50)) if mx > 0.50 else 0.0
    else:              # ContraProST: floor at 50% chance level
        scheme, display = "pink",   f"{v * 100:.1f}"
        norm = max(0.0, (v - 0.50) / (mx - 0.50)) if mx > 0.50 else 0.0
    r, g, b = color_for(norm, scheme)
    sec_best = col_best_per_sec.get(section, [None]*N_SCORE)[col_idx]
    if sec_best is not None and v == sec_best:
        display = rf"\textbf{{{display}}}"
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}{display}"


def delta_cell(d):
    """ASR delta: min→white (worst), max→darkest blue (best/most robust). -- uncoloured."""
    if d is None:
        return "--"
    sign = "+" if d >= 0 else ""
    norm = (d - delta_min) / (delta_max - delta_min) if delta_max > delta_min else 0.5
    r = int(255 + 0.5 * (58  - 255) * norm)
    g = int(255 + 0.5 * (126 - 255) * norm)
    b = int(255 + 0.5 * (192 - 255) * norm)
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}\scriptsize{{{sign}{d:.1f}}}"


# ── build lines ───────────────────────────────────────────────────────────────
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

prev_section = BASELINE
for row in data:
    if row is SEP:
        lines.append(r"\midrule")
        continue
    section = row[0]
    if section != prev_section:
        if section == SCOMET:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{16}{l}{\textit{SpeechCOMET}} \\")
            lines.append(r"\midrule")
        elif section == SLLM:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{16}{l}{\textit{SpeechLLM}} \\")
            lines.append(r"\midrule")
        prev_section = section

    type_col, model = row[1], row[2]
    seg_de, seg_zh, seg_avg, sys_de, sys_zh, sys_avg = row[3:9]
    d_seg, d_sys = row[9], row[10]
    mshe = row[11:14]
    cp   = row[14:17]

    # build score columns (col indices for best-detection)
    s_cols = [seg_de, seg_zh, seg_avg, sys_de, sys_zh, sys_avg] + list(mshe) + list(cp)
    cells = [type_col, model]
    for i, v in enumerate(s_cols[:6]):
        cells.append(score_cell(v, i, section))
        if i == 2:   # after seg_avg, insert Δ
            cells.append(delta_cell(d_seg))
        if i == 5:   # after sys_avg, insert Δ
            cells.append(delta_cell(d_sys))
    for i, v in enumerate(mshe):
        cells.append(score_cell(v, 6 + i, section))
    for i, v in enumerate(cp):
        cells.append(score_cell(v, 9 + i, section))
    lines.append(" & ".join(cells) + r" \\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    (r"\caption{Results for SpeechCOMET and SpeechLLM models."
     r" ASR src shows the change in average $\tau_b$/SPA when using ASR transcripts instead of human transcripts;"
     r" speech-only models have no text input (--)."
     r" $^\dagger$Uses InfoXLM-large as text encoder instead of XLM-RoBERTa-large.}"
     r"\label{tab:main_results_v2}"),
    r"\end{table*}",
]

output = "\n".join(lines)
out_path = os.path.join(os.path.dirname(__file__), "table_main_results_v2.tex")
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"Saved to {out_path}")
