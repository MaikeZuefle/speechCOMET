#!/usr/bin/env python3
"""Random-source analysis table: real source performance and drop when source is shuffled.

Δ = random_src_score − real_src_score (avg over de and zh).
Large negative Δ → model uses the source; Δ ≈ 0 → model ignores it.
"""
import os

# (section, type_label, model_label,
#  seg_real_avg, seg_rand_avg,
#  sys_real_avg, sys_rand_avg)

BASELINE = "Baseline"
SCOMET   = "SpeechCOMET"
SLLM     = "SpeechLLM"
SEP      = "---"

data = [
    # ── Baselines ─────────────────────────────────────────────────────────────
    (BASELINE, r"\multirow{2}{*}{\textsc{Text}}",
     r"\textsc{COMETKiwi-Partial}",
     11.95, 7.60,   55.05, 61.40),
    (BASELINE, "",
     r"\textsc{COMETKiwi}",
     34.60, 9.70,   87.70, 50.25),
    (BASELINE, r"\multirow{2}{*}{\textsc{Speech}}",
     r"\textsc{SpeechQE}",
     29.80, 7.35,   75.25, 51.15),
    (BASELINE, "",
     r"\textsc{BLASER}",
     24.42, 9.05,   76.46, 73.70),
    SEP,
    # ── SpeechCOMET ───────────────────────────────────────────────────────────
    (SCOMET, r"\multirow{2}{*}{\textsc{Text}}",
     r"COMETKiwi$_{\text{RoBERTa}}^{\text{WMT}}$",
     23.10, 12.00,  56.04, 55.55),
    (SCOMET, "",
     r"COMETKiwi$_{\text{RoBERTa}}^{\text{IWSLT}}$",
     22.75, 10.75,  68.67, 72.55),
    (SCOMET, r"\multirow{2}{*}{\textsc{Speech}}",
     r"SpeechCOMET\textsubscript{SONAR}",
     20.10, 17.00,  60.98, 65.15),
    (SCOMET, "",
     r"SpeechCOMET\textsubscript{Whisper}",
     17.78, 15.35,  56.71, 64.55),
    (SCOMET, r"\textsc{Sp.+Txt}",
     r"SpeechCOMET",
     25.80, 8.05,   73.55, 59.00),
    (SCOMET, r"\textsc{Sp.+Txt}",
     r"SpeechCOMET$^\dagger$",
     34.35, 4.20,   76.50, 43.00),
    SEP,
    # ── SpeechLLM ─────────────────────────────────────────────────────────────
    (SLLM, r"\multirow{2}{*}{\textsc{Text}}",
     r"SpeechLLM",
     40.35, 11.15,  64.10, 80.05),
    (SLLM, "",
     r"\quad+FT",
     47.15, 8.50,   88.55, 78.90),
    (SLLM, r"\multirow{2}{*}{\textsc{Speech}}",
     r"SpeechLLM",
     32.00, 3.85,   60.10, 64.20),
    (SLLM, "",
     r"\quad+FT",
     40.12, 14.65,  70.14, 72.60),
    (SLLM, r"\multirow{2}{*}{\textsc{Sp.+Txt}}",
     r"SpeechLLM",
     38.15, 0.70,   61.65, 50.55),
    (SLLM, "",
     r"\quad+FT",
     49.90, 10.70,  80.60, 65.75),
]

# ── colour helpers ─────────────────────────────────────────────────────────────
def score_cell(v, all_vals):
    mn, mx = min(all_vals), max(all_vals)
    norm = (v - mn) / (mx - mn) if mx > mn else 0.5
    r = int(255 + 0.60 * (136 - 255) * norm)
    g = int(255 + 0.60 * (204 - 255) * norm)
    b = int(255 + 0.60 * (203 - 255) * norm)
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}{v:.1f}"


# random-src delta range: min (biggest drop) → white, max (smallest drop) → darkest
all_rand_deltas = [row[4] - row[3] for row in data if row is not SEP]
rand_delta_min = min(all_rand_deltas)
rand_delta_max = max(all_rand_deltas)


def delta_cell(d):
    """min (biggest drop) → darkest blue, max (smallest drop) → white."""
    sign = "+" if d >= 0 else ""
    norm = 1.0 - (d - rand_delta_min) / (rand_delta_max - rand_delta_min) \
           if rand_delta_max > rand_delta_min else 0.5
    r = int(255 + 0.5 * (58  - 255) * norm)
    g = int(255 + 0.5 * (126 - 255) * norm)
    b = int(255 + 0.5 * (192 - 255) * norm)
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}{sign}{d:.1f}"


# normalisation ranges
seg_real = [r[3] for r in data if r is not SEP]
sys_real = [r[5] for r in data if r is not SEP]

SECTION_LABELS = {
    SCOMET: r"\textit{SpeechCOMET}",
    SLLM:   r"\textit{SpeechLLM}",
}

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{ll|rc}",
    r"\toprule",
    (r" & & \multicolumn{2}{c}{\textbf{Segment $\tau_b$ (\%)}} \\"),
    r"\cmidrule(lr){3-4}",
    (r" & & \textbf{real src} & \textbf{$\Delta$ random} \\"),
    r"\midrule",
    r"\multicolumn{4}{l}{\textit{Baselines}} \\",
    r"\midrule",
]

prev_section = BASELINE
for row in data:
    if row is SEP:
        lines.append(r"\midrule")
        continue
    section = row[0]
    if section != prev_section:
        lines.append(r"\multicolumn{4}{l}{" + SECTION_LABELS[section] + r"} \\")
        lines.append(r"\midrule")
        prev_section = section

    type_col, model = row[1], row[2]
    seg_r, seg_rnd, sys_r, sys_rnd = row[3], row[4], row[5], row[6]
    d_seg = seg_rnd - seg_r
    d_sys = sys_rnd - sys_r

    cells = [type_col, model,
             score_cell(seg_r, seg_real), delta_cell(d_seg)]
    lines.append(" & ".join(cells) + r" \\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    (r"\caption{Effect of replacing the source input with a randomly mismatched one."
     r" Scores are averaged over de and zh."
     r" $\Delta$ = random$-$real: large negative values indicate the model uses"
     r" the source signal; $\Delta \approx 0$ indicates the model ignores it."
     r" $^\dagger$Uses InfoXLM-large as text encoder.}"
     r"\label{tab:random_src}"),
    r"\end{table}",
]

output = "\n".join(lines)
out_path = os.path.join(os.path.dirname(__file__), "table_random_src.tex")
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"Saved to {out_path}")
