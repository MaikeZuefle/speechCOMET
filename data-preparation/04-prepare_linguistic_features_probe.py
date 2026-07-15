"""
Add sentence-level morphological features to probe JSONL files.

Features extracted from the source sentence:
  - VerbForm, Tense, Mood  — from the ROOT verb
  - SubjNumber, ObjNumber  — Number (Sing/Plur) of the first nsubj and obj/dobj dependent

Output columns: src_VerbForm, src_Tense, src_Mood, src_SubjNumber, src_ObjNumber.

Usage:
    python add_morph_features.py train_probe.jsonl [--output out.jsonl]
"""

import argparse, json, sys
from pathlib import Path
import spacy
from tqdm import tqdm

try:
    import cupy  # noqa: F401
    if spacy.prefer_gpu():
        print("spaCy is using GPU.", file=sys.stderr)
    else:
        print("WARNING: GPU not available, spaCy will run on CPU.", file=sys.stderr)
except ImportError:
    print("WARNING: cupy not installed, spaCy will run on CPU. "
          "Install with: pip install cupy-cuda12x", file=sys.stderr)

VERB_FEATURES = ["VerbForm", "Tense", "Mood"]
LANG_MODEL = {"en": "en_core_web_trf", "de": "de_core_news_lg"}
# cs_core_news_sm not available for spaCy 3.8.x — Czech entries will get null features

ALL_FEATURES = VERB_FEATURES + ["SubjNumber", "ObjNumber", "HasEntity", "EntityTypes", "Entities"]


def get_nlp(lang, cache={}):
    if lang not in cache:
        model = LANG_MODEL.get(lang)
        if not model:
            print(f"WARNING: no spaCy model configured for '{lang}', features will be null.", file=sys.stderr)
            cache[lang] = None
        else:
            try:
                cache[lang] = spacy.load(model)
            except OSError:
                sys.exit(f"Model '{model}' not found. Run: python -m spacy download {model}")
    return cache[lang]


def extract_features(doc):
    feats = {f: None for f in ALL_FEATURES}

    for tok in doc:
        if tok.dep_ == "ROOT":
            for f in VERB_FEATURES:
                feats[f] = (tok.morph.get(f) or [None])[0]

        if tok.dep_ in ("nsubj", "nsubjpass") and feats["SubjNumber"] is None:
            feats["SubjNumber"] = (tok.morph.get("Number") or [None])[0]

        if tok.dep_ in ("obj", "dobj") and feats["ObjNumber"] is None:
            feats["ObjNumber"] = (tok.morph.get("Number") or [None])[0]

    feats["HasEntity"] = len(doc.ents) > 0
    feats["EntityTypes"] = ",".join(sorted(set(e.label_ for e in doc.ents))) or None
    feats["Entities"] = {e.text: e.label_ for e in doc.ents} or None
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output", default=None)
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="spaCy pipe batch size. Default: 256.")
    args = ap.parse_args()

    out_path = args.output or args.input
    entries = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]

    from collections import defaultdict
    value_counts = defaultdict(lambda: defaultdict(int))

    # Group indices by language so we can use nlp.pipe() per language
    from itertools import groupby
    lang_to_indices = defaultdict(list)
    for i, entry in enumerate(entries):
        lang_to_indices[entry["src_lang"]].append(i)

    # Pre-allocate features for all entries
    all_feats = [None] * len(entries)

    for lang, indices in lang_to_indices.items():
        nlp = get_nlp(lang)
        if nlp is None:
            for i in indices:
                all_feats[i] = {f: None for f in ALL_FEATURES}
        else:
            texts = [entries[i]["src"] for i in indices]
            docs = nlp.pipe(texts, batch_size=args.batch_size)
            for i, doc in zip(indices, tqdm(docs, total=len(indices),
                                            desc=f"Extracting [{lang}]",
                                            dynamic_ncols=True)):
                all_feats[i] = extract_features(doc)

    with open(out_path, "w") as f:
        for entry, feats in zip(entries, all_feats):
            for feat, val in feats.items():
                entry[f"src_{feat}"] = val
                if feat != "Entities":
                    value_counts[feat][val] += 1
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    for feat in ALL_FEATURES:
        print(f"\nsrc_{feat}:")
        for val, count in sorted(value_counts[feat].items(), key=lambda x: -x[1]):
            print(f"  {val}: {count}")
    print(f"\nWrote {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
