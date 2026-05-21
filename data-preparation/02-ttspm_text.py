"""
Generate TTS audio for text_train.csv and text_dev.csv using Kokoro,
then push to HuggingFace as maikezu/ben_nevis_tts.

Columns match maikezu/iwslt2026-metrics-shared-train-dev:
  src_text, tgt_text, audio, score

train.csv → "train" split, dev.csv → "validation" split.
Within each split, TTS is deduplicated (same src → same audio).

TTS backends:
- Kokoro for English, Japanese, Chinese, Spanish, French, Hindi, Italian, Portuguese
- MMS-TTS (facebook/mms-tts-{lang}) as fallback for everything else
- Rows where neither backend works are skipped

Storage: parquet shards instead of individual WAVs to stay within
cluster file limits.
"""

import hashlib
import os

import numpy as np
import pandas as pd
import scipy.signal
import torch
from datasets import Audio, Dataset, Features, Value, load_dataset
from kokoro import KPipeline
from lingua import Language, LanguageDetectorBuilder
from tqdm import tqdm
from transformers import AutoTokenizer, VitsModel

# ── Kokoro language support ────────────────────────────────────────────────────
LANG_VOICES = {
    Language.ENGLISH:    ('a', ["af_heart", "af_alloy", "af_aoede", "af_bella",
                                "af_jessica", "af_kore", "af_nicole", "af_nova",
                                "af_river", "af_sarah", "af_sky", "am_adam",
                                "am_echo", "am_eric", "am_fenrir", "am_liam",
                                "am_michael", "am_onyx", "am_puck",
                                "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
                                "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]),
    Language.JAPANESE:   ('j', ["jf_alpha", "jf_gongitsune", "jf_nezumi",
                                "jf_tebukuro", "jm_kumo"]),
    Language.CHINESE:    ('z', ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao",
                                "zf_xiaoyi", "zm_yunjian", "zm_yunxi",
                                "zm_yunxia", "zm_yunyang"]),
    Language.SPANISH:    ('e', ["ef_dora", "em_alex", "em_santa"]),
    Language.FRENCH:     ('f', ["ff_siwis"]),
    Language.HINDI:      ('h', ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"]),
    Language.ITALIAN:    ('i', ["if_sara", "im_nicola"]),
    Language.PORTUGUESE: ('p', ["pf_dora", "pm_alex", "pm_santa"]),
}

# ── config ─────────────────────────────────────────────────────────────────────
TRAIN_CSV    = "data/train/text_train.csv"
DEV_CSV      = "data/dev/text_dev.csv"
HF_REPO      = "maikezu/ben_nevis_tts"
SHARD_DIR    = "data/tts_shards"
SAMPLE_RATE  = 24000
BATCH_SIZE   = 1000

# ── MMS-TTS fallback (lazy-loaded per language) ────────────────────────────────
_mms_cache = {}  # iso_code -> (model, tokenizer, native_sr) or None


def tts_mms(src_text: str, language: Language):
    iso = language.iso_code_639_3.name.lower()
    if iso not in _mms_cache:
        model_id = f"facebook/mms-tts-{iso}"
        try:
            model = VitsModel.from_pretrained(model_id).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            _mms_cache[iso] = (model, tokenizer, model.config.sampling_rate)
            print(f"  Loaded MMS fallback for '{iso}'")
        except Exception as e:
            print(f"  No MMS model for '{iso}': {e}")
            _mms_cache[iso] = None

    if _mms_cache[iso] is None:
        return None

    model, tokenizer, native_sr = _mms_cache[iso]
    inputs = tokenizer(src_text, return_tensors="pt")
    with torch.no_grad():
        audio = model(**inputs).waveform.squeeze().numpy()

    if native_sr != SAMPLE_RATE:
        n = int(len(audio) * SAMPLE_RATE / native_sr)
        audio = scipy.signal.resample(audio, n)
    return audio


# ── helpers ────────────────────────────────────────────────────────────────────
def pick_kokoro_voice(src_text: str, language: Language):
    lang_code, voices = LANG_VOICES[language]
    h = int(hashlib.md5(src_text.encode()).hexdigest(), 16)
    return lang_code, voices[h % len(voices)]


def load_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["src", "mt"])
    df = df[(df["src"].astype(str).str.strip() != "") & (df["mt"].astype(str).str.strip() != "")]
    df["src"] = df["src"].astype(str)
    df["mt"]  = df["mt"].astype(str)
    return df


def process_and_push(df, split_name, kokoro_pipelines, detector):
    shard_dir = os.path.join(SHARD_DIR, split_name)
    os.makedirs(shard_dir, exist_ok=True)

    src_to_rows = {}
    for _, row in df.iterrows():
        src_to_rows.setdefault(row["src"], []).append(row)

    unique_srcs = list(src_to_rows.keys())
    batches = [unique_srcs[i:i + BATCH_SIZE] for i in range(0, len(unique_srcs), BATCH_SIZE)]
    n_skipped = 0

    for shard_idx, batch_srcs in enumerate(tqdm(batches, desc=f"Shards ({split_name})")):
        path = os.path.join(shard_dir, f"shard_{shard_idx:04d}.parquet")
        if os.path.exists(path):
            continue

        records = []
        for src_text in tqdm(batch_srcs, desc="  TTS", leave=False):
            language = detector.detect_language_of(src_text)

            if language in LANG_VOICES and LANG_VOICES[language][0] in kokoro_pipelines:
                lang_code, voice = pick_kokoro_voice(src_text, language)
                chunks = [a for _, _, a in kokoro_pipelines[lang_code](src_text, voice=voice)]
                audio_array = np.concatenate(chunks) if chunks else None
            else:
                audio_array = tts_mms(src_text, language) if language is not None else None

            if audio_array is None or len(audio_array) == 0:
                n_skipped += 1
                continue

            for row in src_to_rows[src_text]:
                records.append({
                    "src_text": row["src"],
                    "tgt_text": row["mt"],
                    "score":    float(row["score"]),
                    "audio":    {"array": audio_array, "sampling_rate": SAMPLE_RATE},
                })

        if not records:
            continue

        features = Features({
            "src_text": Value("string"),
            "tgt_text": Value("string"),
            "score":    Value("float64"),
            "audio":    Audio(sampling_rate=SAMPLE_RATE),
        })
        shard = Dataset.from_list(records, features=features)
        shard.to_parquet(path)

    print(f"  Skipped {n_skipped} sources (no TTS available)")

    shard_files = sorted(
        os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".parquet")
    )
    dataset = load_dataset("parquet", data_files=shard_files)["train"]
    print(f"Pushing {len(dataset)} rows as split='{split_name}'...")
    dataset.push_to_hub(HF_REPO, split=split_name)


# ── main ───────────────────────────────────────────────────────────────────────
# Build language detector covering all lingua languages (MMS covers the rest)
detector = LanguageDetectorBuilder.from_all_languages().build()

# Load Kokoro pipelines, skip any with missing dependencies
kokoro_pipelines = {}
for lc in {v[0] for v in LANG_VOICES.values()}:
    try:
        kokoro_pipelines[lc] = KPipeline(lang_code=lc)
    except Exception as e:
        print(f"  Skipping Kokoro pipeline '{lc}': {e}")
print(f"Loaded Kokoro pipelines: {sorted(kokoro_pipelines.keys())}")

train_df = load_csv(TRAIN_CSV)
dev_df   = load_csv(DEV_CSV)
print(f"Loaded {len(train_df)} train + {len(dev_df)} dev rows")

process_and_push(train_df, "train",      kokoro_pipelines, detector)
process_and_push(dev_df,   "validation", kokoro_pipelines, detector)

print(f"Done! https://huggingface.co/datasets/{HF_REPO}")
