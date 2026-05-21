#!/usr/bin/env python3
"""
Update the four data/results/ display CSVs with fresh analysis results.

Sources:
  data/wer_analysis/wer_correlation_results.csv          → Results.csv (ASR rows)
  data/wer_analysis/wer_correlation_challenge80_results.csv → HighWER.csv
  data/contraprost_analysis/combined_results.csv          → MustSHE.csv
  trained_models/*/shuffled_src/correlation_*.txt         → RandomSrc.csv

Only SpeechCOMET trained-model rows are updated; baselines and SpeechLLM rows
are left unchanged.
"""
import csv
import os
import re

# ── Model name mapping: folder name → display name ───────────────────────────
# For models that share a display name, also provide a disambiguator that must
# appear (or not) in the Encoder column of the display CSV (col 3).
MODEL_MAP = {
    # (folder_name): (display_name, encoder_contains_or_None)
    "lewis-10ep":                               ("Lewis",                   None),
    "lewis-BIG":                                ("Lewis-BIG",               None),
    "skye-20ep":                                ("Skye",                    None),
    "harris-20ep":                              ("Harris",                  None),
    "harris-FT-sonar":                          ("Harris-FT-sonar",         None),
    "shetland-20ep":                            ("Shetland",                None),
    "shetland-FT-sonar":                        ("Shetland-FT-sonar",       None),
    "bute-pretrain":                            ("Lewis-TTS",               None),
    "bute-pretrain-unfreeze-sonar":             ("Lewis TTS-unfreeze",      None),
    "bute-train":                               ("Lewis-TTS+Harris",        None),
    "bute-pretrain-unfreeze-sonar-train-freeze":("Lewis-TTS-unfreeze+Harris", "unfreeze SONAR", False),
    "bute-pretrain-unfreeze-sonar-train-unfreeze":("Lewis-TTS-unfreeze+Harris","unfreeze SONAR", True),
    "bute-pretrain-whisper-attn-pool":          ("Whisper-Lewis-TTS",       None),
    "mull-attn-from-ckpt":                      ("Whisper-Lewis-TTS +Harris","LORA Whisper", False),
    "mull-attn-lora-from-ckpt":                 ("Whisper-Lewis-TTS +Harris","LORA Whisper", True),
    "mull-avg-20ep":                            ("Mull-avg",               "LORA Whisper", False),
    "mull-avg-lora-10ep":                       ("Mull-avg",               "LORA Whisper", True),
    "mull-attn-10ep":                           ("Mull-attn-pool",         "LORA Whisper", False),
    "mull-attn-lora-10ep":                      ("Mull-attn-pool",         "LORA Whisper", True),
    "orkney-avg-20ep":                          ("Orkney-avg",              None),
    "orkney-sum-20ep":                          ("Orkney-sum",              None),
    "orkney-concat-20ep":                       ("Orkney-concat",           None),
    "orkney-sum-from-text-ckpt-20ep":           ("Orkney-sum-from-text",    None),
    "orkney-sum-from-text-ckpt-BIG":            ("Orkney-sum-from-text-BIG",None),
    "orkney-sum-from-text-ckpt-FT-sonar":       ("Orkney-sum-from-text-unfreeze", None),
}

# Text models have human + ASR rows; only update the ASR row from wer analysis
TEXT_MODELS = {"Lewis", "Lewis-BIG", "Skye"}


def pct(v):
    """Format float as percentage string, e.g. 0.1234 → '12.34%'."""
    if v is None or v == "":
        return ""
    try:
        return f"{float(v) * 100:.2f}%"
    except (ValueError, TypeError):
        return ""


def find_rows(lines, display_name, encoder_contains=None, encoder_present=None,
              asr_only=False):
    """Return list of (line_index, row) matching the given display name + constraints.

    Handles two patterns of continuation rows (empty model name in col 2):
    - asr_only: look ahead for row with empty name and col 5 == 'ASR'
    - encoder_present=True: look ahead for row with empty name and encoder matching
    If the look-ahead pattern isn't found (e.g. the CSV only has a single named row),
    falls back to matching the named row directly.
    """
    matches = []
    for i, line in enumerate(lines):
        row = next(csv.reader([line]))
        if len(row) < 3:
            continue
        if row[2].strip() != display_name:
            continue
        encoder = row[3].strip() if len(row) > 3 else ""

        if asr_only:
            # First try to find an ASR continuation sub-row (empty model name)
            found_sub = False
            for j in range(i + 1, min(i + 4, len(lines))):
                sub = next(csv.reader([lines[j]]))
                if len(sub) < 6:
                    continue
                if sub[2].strip() == "" and sub[5].strip().upper() == "ASR":
                    matches.append((j, sub))
                    found_sub = True
                    break
            if not found_sub:
                # Fall back: this CSV has a single named row with ASR transcript
                transcript = row[5].strip() if len(row) > 5 else ""
                if transcript.upper() == "ASR":
                    matches.append((i, row))
        elif encoder_contains is not None and encoder_present is True:
            # Want sub-row where encoder CONTAINS the key
            if encoder_contains in encoder:
                matches.append((i, row))
            else:
                # Look ahead for a continuation row with matching encoder
                for j in range(i + 1, min(i + 4, len(lines))):
                    sub = next(csv.reader([lines[j]]))
                    if sub[2].strip() != "":
                        break  # hit another named row
                    sub_enc = sub[3].strip() if len(sub) > 3 else ""
                    if encoder_contains in sub_enc:
                        matches.append((j, sub))
                        break
        elif encoder_contains is not None and encoder_present is False:
            # Want named row where encoder does NOT contain the key
            if encoder_contains not in encoder:
                matches.append((i, row))
        else:
            matches.append((i, row))
    return matches


def write_row(lines, i, row):
    """Replace line i with the updated row."""
    lines[i] = ",".join(row) + "\n"


def pad(row, length):
    """Extend row to at least `length` columns."""
    while len(row) < length:
        row.append("")
    return row


# ── Load analysis CSVs ───────────────────────────────────────────────────────

def load_csv_dict(path, key_col=0):
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row[reader.fieldnames[key_col]].strip()
            result[key] = row
    return result


wer_corr   = load_csv_dict("data/wer_analysis/wer_correlation_results.csv")
wer_ch80   = load_csv_dict("data/wer_analysis/wer_correlation_challenge80_results.csv")
contraprost = load_csv_dict("data/contraprost_analysis/combined_results.csv")


def load_shuffled_src(folder):
    """Read per-language shuffled source τ_b and SPA from correlation txt files."""
    result = {}
    base = os.path.join("trained_models", folder, "shuffled_src")
    for lang in ("en-de", "en-zh"):
        path = os.path.join(base, f"correlation_dev_asr_{lang}.txt")
        if not os.path.exists(path):
            continue
        text = open(path).read()
        seg = re.search(r"SEGMENT-LEVEL.*?:\s*([\d.]+)%", text, re.S)
        sys = re.search(r"SYSTEM-LEVEL.*?:\s*([\d.]+)%", text, re.S)
        result[lang] = {
            "seg": float(seg.group(1)) / 100 if seg else None,
            "sys": float(sys.group(1)) / 100 if sys else None,
        }
    return result


# ── Column mappings ───────────────────────────────────────────────────────────

WER_BINS = [
    "segment_de_≤0.1","segment_de_0.1–0.3","segment_de_0.3–0.5",
    "segment_de_0.5–0.7","segment_de_0.7–1.0","segment_de_>1.0",
    "segment_zh_≤0.1","segment_zh_0.1–0.3","segment_zh_0.3–0.5",
    "segment_zh_0.5–0.7","segment_zh_0.7–1.0","segment_zh_>1.0",
    "system_de_≤0.1","system_de_0.1–0.3","system_de_0.3–0.5",
    "system_de_0.5–0.7","system_de_0.7–1.0","system_de_>1.0",
    "system_zh_≤0.1","system_zh_0.1–0.3","system_zh_0.3–0.5",
    "system_zh_0.5–0.7","system_zh_0.7–1.0","system_zh_>1.0",
]

MUSTSHE_COLS = [
    "mustshe_es","mustshe_fr","mustshe_it",
    "contraprost_de","contraprost_es","contraprost_ja",
    "mustshe_es_1F","mustshe_es_1M",
    "mustshe_fr_1F","mustshe_fr_1M",
    "mustshe_it_1F","mustshe_it_1M",
    "contraprost_de_Stress","contraprost_de_Breaks","contraprost_de_Intonation",
    "contraprost_de_Emotion","contraprost_de_Politeness",
    "contraprost_es_Stress","contraprost_es_Breaks","contraprost_es_Intonation",
    "contraprost_es_Emotion","contraprost_es_Politeness",
    "contraprost_ja_Stress","contraprost_ja_Breaks","contraprost_ja_Intonation",
    "contraprost_ja_Emotion","contraprost_ja_Politeness",
]


# ── Update functions ──────────────────────────────────────────────────────────

def update_results(lines, display_name, folder, entry, asr_only,
                   enc_contains=None, enc_present=None):
    matches = find_rows(lines, display_name, enc_contains, enc_present, asr_only=asr_only)
    if not matches:
        print(f"  WARNING: no row found for '{display_name}' (asr_only={asr_only})")
        return
    i, row = matches[0]
    row = pad(row, 41)
    row[8]  = pct(entry.get("segment_de"))
    row[9]  = pct(entry.get("segment_zh"))
    row[10] = pct(entry.get("segment_avg"))
    row[11] = pct(entry.get("system_de"))
    row[12] = pct(entry.get("system_zh"))
    row[13] = pct(entry.get("system_avg"))
    row[14] = pct(entry.get("wer_r_de"))
    row[15] = pct(entry.get("wer_r_zh"))
    row[16] = pct(entry.get("wer_r_avg"))
    for j, col in enumerate(WER_BINS):
        row[17 + j] = pct(entry.get(col))
    write_row(lines, i, row)
    print(f"  Updated Results: {display_name}")


def update_highwer(lines, display_name, folder, entry_ch80, entry_wer,
                   encoder_contains=None, encoder_present=None):
    matches = find_rows(lines, display_name, encoder_contains, encoder_present)
    if not matches:
        print(f"  WARNING: no row found for '{display_name}' in HighWER")
        return
    i, row = matches[0]
    row = pad(row, 59)
    # cols 8-31: challenge80 WER bins
    for j, col in enumerate(WER_BINS):
        row[8 + j] = pct(entry_ch80.get(col))
    # cols 35-58: regular WER bins (same bin names)
    for j, col in enumerate(WER_BINS):
        row[35 + j] = pct(entry_wer.get(col))
    write_row(lines, i, row)
    print(f"  Updated HighWER:  {display_name}")


def update_mustshe(lines, display_name, folder, entry,
                   encoder_contains=None, encoder_present=None):
    matches = find_rows(lines, display_name, encoder_contains, encoder_present)
    if not matches:
        print(f"  WARNING: no row found for '{display_name}' in MustSHE")
        return
    i, row = matches[0]
    row = pad(row, 34)
    for j, col in enumerate(MUSTSHE_COLS):
        val = entry.get(col, "")
        try:
            row[7 + j] = f"{float(val):.3f}" if val else ""
        except (ValueError, TypeError):
            row[7 + j] = ""
    write_row(lines, i, row)
    print(f"  Updated MustSHE:  {display_name}")


def update_randomsrc(lines, display_name, folder):
    shuf = load_shuffled_src(folder)
    if not shuf:
        print(f"  WARNING: no shuffled_src data for {folder}")
        return
    matches = find_rows(lines, display_name)
    if not matches:
        print(f"  WARNING: no row found for '{display_name}' in RandomSrc")
        return
    i, row = matches[0]
    row = pad(row, 14)
    de = shuf.get("en-de", {})
    zh = shuf.get("en-zh", {})
    seg_de = de.get("seg"); seg_zh = zh.get("seg")
    sys_de = de.get("sys"); sys_zh = zh.get("sys")
    seg_avg = (seg_de + seg_zh) / 2 if seg_de is not None and seg_zh is not None else None
    sys_avg = (sys_de + sys_zh) / 2 if sys_de is not None and sys_zh is not None else None
    row[8]  = pct(seg_de)
    row[9]  = pct(seg_zh)
    row[10] = pct(seg_avg)
    row[11] = pct(sys_de)
    row[12] = pct(sys_zh)
    row[13] = pct(sys_avg)
    write_row(lines, i, row)
    print(f"  Updated RandomSrc:{display_name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    res_path  = "data/results/SpeechCOMET experiments - Results.csv"
    hwer_path = "data/results/SpeechCOMET experiments - High WER + Good translation.csv"
    msh_path  = "data/results/SpeechCOMET experiments - MustSHE + ContrProsT.csv"
    rnd_path  = "data/results/SpeechCOMET experiments - Random Source.csv"

    res_lines  = open(res_path).readlines()
    hwer_lines = open(hwer_path).readlines()
    msh_lines  = open(msh_path).readlines()
    rnd_lines  = open(rnd_path).readlines()

    for folder, info in MODEL_MAP.items():
        if len(info) == 2:
            display_name, _ = info
            enc_contains = enc_present = None
        else:
            display_name, enc_contains, enc_present = info

        asr_only = display_name in TEXT_MODELS

        print(f"\n[{folder}]")

        # ── Results.csv ──────────────────────────────────────────────────────
        if folder in wer_corr:
            update_results(res_lines, display_name, folder,
                           wer_corr[folder], asr_only, enc_contains, enc_present)
        else:
            print(f"  (not in wer_correlation_results)")

        # ── HighWER.csv ──────────────────────────────────────────────────────
        ch80 = wer_ch80.get(folder, {})
        wer  = wer_corr.get(folder, {})
        if ch80 or wer:
            update_highwer(hwer_lines, display_name, folder, ch80, wer,
                           enc_contains, enc_present)

        # ── MustSHE.csv ──────────────────────────────────────────────────────
        if folder in contraprost:
            update_mustshe(msh_lines, display_name, folder,
                           contraprost[folder], enc_contains, enc_present)
        else:
            print(f"  (not in combined_results)")

        # ── RandomSrc.csv ────────────────────────────────────────────────────
        update_randomsrc(rnd_lines, display_name, folder)

    for path, lines in [
        (res_path, res_lines), (hwer_path, hwer_lines),
        (msh_path, msh_lines), (rnd_path, rnd_lines),
    ]:
        with open(path, "w") as f:
            f.writelines(lines)
        print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
