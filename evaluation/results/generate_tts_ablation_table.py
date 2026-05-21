#!/usr/bin/env python3
"""Ablation table: SpeechCOMET pretraining strategies — multirow grouped layout."""
import os

# Groups: (section, base_name, [(variant_label, seg_de, seg_zh, seg_avg, sys_de, sys_zh, sys_avg,
#                                 mshe_es, mshe_fr, mshe_it, cp_de, cp_es, cp_ja), ...])

SONAR   = "SONAR"
WHISPER = "Whisper"
SEP      = "---"   # cmidrule(l){1-14} between main groups
SEP_INNER= "inner" # cmidrule(l){3-14} within a group (skips model col)

groups = [
    # ── SONAR ─────────────────────────────────────────────────────────────────
    (SONAR, "IWSLT FT", [
        ("frozen",
         14.61, 22.46, 18.54,  37.23, 80.40, 58.82,
         0.560, 0.519, 0.513,  0.500, 0.500, 0.500),
        ("unfrozen",
         17.34, 22.86, 20.10,  51.05, 70.90, 60.98,
         0.531, 0.509, 0.509,  0.500, 0.500, 0.500),
    ]),
    SEP,
    (SONAR, r"WMT text pretrain $\to$ IWSLT FT", [
        ("frozen",
         15.59, 19.26, 17.43,  41.12, 68.30, 54.71,
         0.531, 0.530, 0.501,  0.501, 0.500, 0.500),
        ("unfrozen",
         16.69, 21.16, 18.93,  46.25, 73.60, 59.93,
         0.535, 0.523, 0.505,  0.500, 0.500, 0.500),
    ]),
    SEP,
    (SONAR, r"WMT TTS pretrain (frozen)", [
        ("pretrain only",
         10.72, 16.58, 13.65,  23.63, 71.50, 47.57,
         0.537, 0.521, 0.546,  0.499, 0.500, 0.500),
        (r"\quad $\to$ IWSLT FT (frozen)",
         13.79, 18.82, 16.31,  32.68, 67.70, 50.19,
         0.537, 0.523, 0.544,  0.500, 0.499, 0.500),
    ]),
    SEP_INNER,
    (SONAR, r"WMT TTS pretrain (unfrozen)", [
        ("pretrain only",
         11.58, 15.37, 13.48,  26.28, 72.10, 49.19,
         0.509, 0.509, 0.534,  0.500, 0.502, 0.497),
        (r"\quad $\to$ IWSLT FT (frozen)",
         15.07, 20.03, 17.55,  32.68, 67.90, 50.29,
         0.564, 0.517, 0.530,  0.500, 0.500, 0.501),
        (r"\quad $\to$ IWSLT FT (unfrozen)",
         15.67, 21.75, 18.71,  37.68, 68.60, 53.14,
         0.541, 0.540, 0.532,  0.499, 0.500, 0.503),
    ]),
    # ── Whisper ───────────────────────────────────────────────────────────────
    (WHISPER, "IWSLT FT (avg pool)", [
        ("frozen",
         10.57, 18.61, 14.59,  39.65, 67.60, 53.63,
         0.505, 0.469, 0.491,  0.500, 0.499, 0.500),
        ("LoRA",
         7.70,  16.46, 12.08,  35.22, 67.40, 51.31,
         0.541, 0.516, 0.491,  0.501, 0.500, 0.499),
    ]),
    (WHISPER, "IWSLT FT (attn pool)", [
        ("frozen",
         6.58,  16.68, 11.63,  44.40, 67.40, 55.90,
         0.558, 0.509, 0.519,  0.501, 0.499, 0.500),
        ("LoRA",
         7.94,  19.45, 13.70,  42.45, 67.40, 54.93,
         0.556, 0.519, 0.534,  0.500, 0.501, 0.500),
    ]),
    SEP,
    (WHISPER, r"WMT TTS pretrain (LoRA)", [
        ("pretrain only",
         11.88, 13.10, 12.49,  46.08, 59.90, 52.99,
         0.495, 0.533, 0.481,  0.499, 0.502, 0.499),
        (r"\quad $\to$ IWSLT FT (frozen)",
         16.83, 18.72, 17.78,  44.92, 68.50, 56.71,
         0.514, 0.507, 0.528,  0.500, 0.500, 0.501),
        (r"\quad $\to$ IWSLT FT (LoRA)",
         16.09, 19.37, 17.73,  41.65, 68.50, 55.08,
         0.503, 0.505, 0.524,  0.499, 0.499, 0.500),
    ]),
]

N_SCORE = 12
COLORS = {"teal": (136,204,203), "orange": (247,136,50), "pink": (217,110,173)}
INTENSITY = 0.60

def color_for(norm, scheme):
    tr,tg,tb = COLORS[scheme]
    return (int(255+INTENSITY*(tr-255)*norm),
            int(255+INTENSITY*(tg-255)*norm),
            int(255+INTENSITY*(tb-255)*norm))

# collect all rows for global stats
all_rows = [v for g in groups if g is not SEP for v in g[2]]
col_vals = [[] for _ in range(N_SCORE)]
for row in all_rows:
    for i,v in enumerate(list(row[1:7])+list(row[7:13])):
        col_vals[i].append(v)
col_min  = [min(v) for v in col_vals]
col_max  = [max(v) for v in col_vals]

# per-section bests
sec_best = {}
for sec in [SONAR, WHISPER]:
    rows = [v for g in groups if g is not SEP and g[0]==sec for v in g[2]]
    bests = []
    for i in range(N_SCORE):
        bests.append(max((list(r[1:7])+list(r[7:13]))[i] for r in rows))
    sec_best[sec] = bests

def cell(v, col_idx, section):
    mn,mx = col_min[col_idx], col_max[col_idx]
    if col_idx < 6:
        norm = (v-mn)/(mx-mn) if mx>mn else 0.5
        scheme,display = "teal",  f"{v:.1f}"
    elif col_idx < 9:
        norm = max(0.0, (v-0.50)/(mx-0.50)) if mx>0.50 else 0.0
        scheme,display = "orange",f"{v*100:.1f}"
    else:
        norm = max(0.0, (v-0.50)/(mx-0.50)) if mx>0.50 else 0.0
        scheme,display = "pink",  f"{v*100:.1f}"
    r,g,b = color_for(norm,scheme)
    if v == sec_best[section][col_idx]: display = rf"\textbf{{{display}}}"
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}{display}"

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\begin{tabular}{p{3cm}l|rrr|rrr|rrr|rrr}",
    r"\toprule",
    (r" & & \multicolumn{6}{c|}{\textbf{IWSLT dev}}"
     r" & \multicolumn{3}{c|}{\textbf{MuST-SHE}}"
     r" & \multicolumn{3}{c}{\textbf{ContraProST}} \\"),
    (r" & & \multicolumn{3}{c|}{Segment $\tau_b$ (\%)}"
     r" & \multicolumn{3}{c|}{System SPA (\%)}"
     r" & \multicolumn{3}{c|}{PA (\%)}"
     r" & \multicolumn{3}{c}{PA (\%)} \\"),
    r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}",
    (r"\textbf{Model} & \textbf{Variant}"
     r" & \textbf{de} & \textbf{zh} & \textbf{avg}"
     r" & \textbf{de} & \textbf{zh} & \textbf{avg}"
     r" & \textbf{es} & \textbf{fr} & \textbf{it}"
     r" & \textbf{de} & \textbf{es} & \textbf{ja} \\"),
    r"\midrule",
    r"\multicolumn{14}{l}{\textit{SONAR encoder}} \\",
    r"\midrule",
]

prev_section = SONAR
for g in groups:
    if g is SEP:
        lines.append(r"\cmidrule(l){1-14}")
        continue
    if g is SEP_INNER:
        lines.append(r"\cmidrule(l){2-14}")
        continue
    section, base_name, variants = g
    if section != prev_section:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{14}{l}{\textit{Whisper encoder}} \\")
        lines.append(r"\midrule")
        prev_section = section

    n = len(variants)
    for k, row in enumerate(variants):
        variant_label = row[0]
        vals = list(row[1:7]) + list(row[7:13])
        if k == 0:
            if n > 1:
                name_cell = rf"\multirow{{{n}}}{{*}}{{\parbox[t]{{3cm}}{{\raggedright {base_name}}}}}"
            else:
                name_cell = rf"\parbox[t]{{3cm}}{{\raggedright {base_name}}}"
        else:
            name_cell = ""
        cells = [name_cell, variant_label]
        for i, v in enumerate(vals):
            cells.append(cell(v, i, section))
        lines.append(" & ".join(cells) + r" \\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    (r"\caption{Ablation of pretraining strategies for SpeechCOMET."
     r" $\to$ denotes sequential training stages."
     r" TTS pretraining uses WMT data with text-to-speech synthesis.}"
     r"\label{tab:tts_ablation}"),
    r"\end{table*}",
]

output = "\n".join(lines)
out_path = os.path.join(os.path.dirname(__file__), "table_tts_ablation.tex")
with open(out_path, "w") as f:
    f.write(output + "\n")
print(f"Saved to {out_path}")
