#!/usr/bin/env python
"""Probing script for COMET referenceless model embeddings.

Extracts src_sentemb, mt_sentemb, and embedded_sequences from a frozen
COMET or speechCOMET model and trains separate MLP classifiers to predict
a categorical attribute (e.g., tense) from each embedding type.

For speechCOMET models with input_modality='audio' or 'audiotext', source
embeddings are extracted via the SONAR audio encoder (encoder_model_audio)
using waveforms from the 'src_audio' field of the JSONL data.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import classification_report

# Patch for older torchmetrics that lack MulticlassMatthewsCorrCoef
try:
    from torchmetrics.classification import MulticlassMatthewsCorrCoef  # noqa: F401
except ImportError:
    import torchmetrics
    from torchmetrics import MatthewsCorrCoef as _MCC

    class _MulticlassMatthewsCorrCoef(_MCC):
        """Shim: wraps the old MatthewsCorrCoef under the new name."""
        def __init__(self, num_classes, **kwargs):
            super().__init__(num_classes=num_classes, **kwargs)

    torchmetrics.classification.MulticlassMatthewsCorrCoef = _MulticlassMatthewsCorrCoef
    import torchmetrics.classification as _tc
    _tc.MulticlassMatthewsCorrCoef = _MulticlassMatthewsCorrCoef

# Patch for older transformers passing deprecated use_auth_token to hf_hub_download
import huggingface_hub as _hfh
_orig_hf_hub_download = _hfh.hf_hub_download
def _patched_hf_hub_download(*args, **kwargs):
    kwargs.pop("use_auth_token", None)
    return _orig_hf_hub_download(*args, **kwargs)
_hfh.hf_hub_download = _patched_hf_hub_download
# Also patch the reference used inside transformers
try:
    import transformers.utils.hub as _thub
    _thub.hf_hub_download = _patched_hf_hub_download
except Exception:
    pass

from comet.models import download_model
from comet.models import load_from_checkpoint as _comet_load_from_checkpoint


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str, attributes: list, require_audio: bool = False):
    """Load and validate JSONL data. Each line must have src, mt, and all attribute columns.

    If require_audio is True, each line must also have a 'src_audio' field.
    """
    data = []
    required = ["src", "mt"] + attributes
    if require_audio:
        required.append("src_audio")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for key in required:
                if key not in obj:
                    raise ValueError(f"Line {i + 1}: missing required field '{key}'")
            data.append(obj)
    if not data:
        raise ValueError(f"No data found in {path}")
    return data


def build_label_mapping(data, attribute: str):
    """Map attribute strings → integer labels. Returns label_to_id and id_to_label.
    None values are mapped to the string 'None' for sorting stability.
    """
    labels = sorted(set(str(d[attribute]) for d in data))
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(model, texts, batch_size, device):
    """Extract sentence embeddings from the COMET encoder.

    Follows the pattern in comet/cli/mbr.py:build_embeddings().
    """
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    inputs = [model.encoder.prepare_sample(batch) for batch in batches]

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(inputs, desc="  Encoding", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emb = model.get_sentence_embedding(input_ids, attention_mask)
            embeddings.append(emb.cpu())

    return torch.vstack(embeddings)


def _cache_key(data_path: str, model_name: str, modality: str) -> str:
    """Deterministic hash for caching embeddings."""
    with open(data_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    name_hash = hashlib.md5(model_name.encode()).hexdigest()
    return f"{file_hash}_{name_hash}_{modality}"


def extract_audio_embeddings(model, audio_items, batch_size, device):
    """Extract source embeddings from the SONAR audio encoder.

    audio_items is a list of wav file paths (strings) or HF AudioDecoder objects,
    as stored in the 'src_audio' field of the JSONL data.
    """
    batches = [audio_items[i : i + batch_size] for i in range(0, len(audio_items), batch_size)]
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(batches, desc="  Encoding audio", dynamic_ncols=True):
            prepared = model.encoder_model_audio.prepare_sample(batch)
            waveforms = prepared["waveforms"]
            emb = model.get_audio_embedding(waveforms)
            embeddings.append(emb.cpu())
    return torch.vstack(embeddings)


def extract_all_embeddings(model, data, data_path, args, modality):
    """Extract src, mt, and embedded_sequences embeddings. Optionally cache as .pt files.

    For modality='text':
        src_emb comes from the text encoder.
    For modality='audio':
        src_emb comes from the SONAR audio encoder (encoder_model_audio).
    For modality='audiotext':
        src_emb is a fusion of text and audio embeddings, replicating
        SpeechRegression.forward() fusion logic.

    embedded_sequences replicates the model forward pass:
        cat(mt_sentemb, src_sentemb, mt_sentemb * src_sentemb, |mt_sentemb - src_sentemb|)

    Returns dict with keys: src_emb, mt_emb, embedded_sequences.
    """
    device = args.device

    # Check cache
    cache_path = None
    if args.embeddings_dir:
        os.makedirs(args.embeddings_dir, exist_ok=True)
        # When train_frac < 1 each seed samples a different subset; key on seed
        # for the train split to avoid embedding/label mismatch across seeds.
        tag = modality + (f"_seed{args.seed}" if data_path == args.train_data and args.train_frac < 1.0 else "")
        key = _cache_key(data_path, args.model, tag)
        cache_path = os.path.join(args.embeddings_dir, f"{key}.pt")
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            return torch.load(cache_path, weights_only=True)

    mts = [d["mt"] for d in data]

    # --- Source embeddings ---
    if modality == "audio":
        audio_items = [d[args.src_audio_field] for d in data]
        print("Extracting source embeddings (audio encoder)...")
        src_emb = extract_audio_embeddings(model, audio_items, args.batch_size, device)
    elif modality == "audiotext":
        sources = [d["src"] for d in data]
        audio_items = [d[args.src_audio_field] for d in data]
        print("Extracting source text embeddings (for audiotext fusion)...")
        text_emb = extract_embeddings(model, sources, args.batch_size, device)
        print("Extracting source audio embeddings (for audiotext fusion)...")
        audio_emb = extract_audio_embeddings(model, audio_items, args.batch_size, device)
        print("Fusing text and audio source embeddings...")
        fuse = model.fuse_emb_strategy
        if fuse == "concat":
            # fusion_proj lives on the model device; run on GPU then move back
            text_emb_d = text_emb.to(device)
            audio_emb_d = audio_emb.to(device)
            with torch.no_grad():
                src_emb = model.fusion_proj(torch.cat([text_emb_d, audio_emb_d], dim=-1)).cpu()
        elif fuse == "sum":
            text_emb_d = text_emb.to(device)
            audio_emb_d = audio_emb.to(device)
            with torch.no_grad():
                src_emb = model.fusion_layernorm(text_emb_d + audio_emb_d).cpu()
        else:  # avg
            src_emb = (text_emb + audio_emb) / 2
    else:  # text
        sources = [d["src"] for d in data]
        print("Extracting source embeddings (text encoder)...")
        src_emb = extract_embeddings(model, sources, args.batch_size, device)

    # --- MT embeddings (always text encoder) ---
    print("Extracting MT embeddings...")
    mt_emb = extract_embeddings(model, mts, args.batch_size, device)

    print("Computing embedded_sequences (forward cat)...")
    diff_src = torch.abs(mt_emb - src_emb)
    prod_src = mt_emb * src_emb
    embedded_sequences = torch.cat((mt_emb, src_emb, prod_src, diff_src), dim=1)

    result = {
        "src_emb": src_emb,
        "mt_emb": mt_emb,
        "embedded_sequences": embedded_sequences,
    }

    if cache_path:
        torch.save(result, cache_path)
        print(f"Cached embeddings to {cache_path}")

    return result


# ---------------------------------------------------------------------------
# MLP Probe
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    """1-N hidden-layer MLP classifier for probing."""

    def __init__(self, input_dim, num_classes, hidden_dims=(256,), dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EmbeddingDataset(Dataset):
    """Wraps (embedding, label) pairs for DataLoader."""

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe(train_embeddings, train_labels, val_embeddings, val_labels, id_to_label, emb_name, args):
    """Train an MLP probe on the given embeddings and labels.

    Returns a dict with val accuracy and classification report string.
    """
    num_classes = len(id_to_label)
    input_dim = train_embeddings.shape[1]

    train_ds = EmbeddingDataset(train_embeddings, train_labels)
    val_ds = EmbeddingDataset(val_embeddings, val_labels)
    train_loader = DataLoader(train_ds, batch_size=args.probe_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.probe_batch_size)

    # Build probe
    probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = MLPProbe(
        input_dim, num_classes,
        hidden_dims=args.hidden_dims, dropout=args.dropout,
    ).to(probe_device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        probe.train()
        total_loss, correct, total = 0.0, 0, 0
        for emb_batch, lbl_batch in train_loader:
            emb_batch = emb_batch.to(probe_device)
            lbl_batch = lbl_batch.to(probe_device)
            logits = probe(emb_batch)
            loss = criterion(logits, lbl_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * lbl_batch.size(0)
            correct += (logits.argmax(dim=1) == lbl_batch).sum().item()
            total += lbl_batch.size(0)
        train_acc = correct / total

        # --- Validate ---
        probe.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for emb_batch, lbl_batch in val_loader:
                emb_batch = emb_batch.to(probe_device)
                lbl_batch = lbl_batch.to(probe_device)
                logits = probe(emb_batch)
                val_correct += (logits.argmax(dim=1) == lbl_batch).sum().item()
                val_total += lbl_batch.size(0)
        val_acc = val_correct / val_total

        print(
            f"  [{emb_name}] Epoch {epoch:>3}/{args.epochs} | "
            f"train_loss={total_loss / total:.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    # Final evaluation with best model
    probe.load_state_dict(best_state)
    probe.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for emb_batch, lbl_batch in val_loader:
            emb_batch = emb_batch.to(probe_device)
            preds = probe(emb_batch).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(lbl_batch)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    target_names = [id_to_label[i] for i in range(num_classes)]
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=target_names,
        digits=4,
    )

    return {"val_acc": best_val_acc, "report": report}


def run_attribute_probes(attributes, train_data, dev_data, train_embs, dev_embs, args):
    """Train a probe for every (attribute, embedding) pair and print a summary.

    train_embs / dev_embs map embedding name (e.g. "src_emb") to a tensor;
    iteration order determines the order probes are trained/reported in.
    Returns the results dict keyed by "{attribute}/{emb_name}".
    """
    results = {}
    for attribute in attributes:
        label_to_id, id_to_label = build_label_mapping(train_data, attribute)
        print(f"Attribute '{attribute}': {len(label_to_id)} classes: {list(label_to_id.keys())}")
        train_labels = torch.tensor([label_to_id[str(d[attribute])] for d in train_data], dtype=torch.long)
        dev_labels = torch.tensor([label_to_id[str(d[attribute])] for d in dev_data], dtype=torch.long)

        for emb_name, train_emb in train_embs.items():
            key = f"{attribute}/{emb_name}"
            print(f"=== Training probe for {key} ===")
            res = train_probe(
                train_emb, train_labels,
                dev_embs[emb_name], dev_labels,
                id_to_label, key, args,
            )
            results[key] = res
            print(f"  Best val accuracy: {res['val_acc']:.4f}")
            print(res["report"])
            print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, res in results.items():
        print(f"  {key:40s}  val_acc = {res['val_acc']:.4f}")
    print("=" * 60)
    return results


# ---------------------------------------------------------------------------
# Qwen2.5-Omni embedding extraction
# ---------------------------------------------------------------------------

# Token id of 'assistant' in the Qwen2.5-Omni tokenizer (verified: id 77091).
_QWEN_ASSISTANT_TOKEN_ID = 77091


def extract_qwen_embeddings(model, processor, data, src_audio_field, system_prompt, layer_idx, probe_prompt=None, modality="audio", last_user_token=False):
    """Extract LLM hidden state at a chosen position for each sample.

    modality controls what goes in the user turn:
      'audio'     — wav only:          [system | user: <audio>]
      'text'      — src text only:     [system | user: <text>]
      'audiotext' — audio then text:   [system | user: <audio> <text>]

    If probe_prompt is None, extracts at the last 'assistant' token (id 77091).
    If probe_prompt is given, it is appended after the generation-prompt suffix
    and the last real token of the resulting input is used instead.
    """
    import soundfile as sf

    try:
        from qwen_omni_utils import process_mm_info
    except ImportError:
        process_mm_info = None

    device = next(model.thinker.parameters()).device
    needs_audio = modality in ("audio", "audiotext")
    embeddings = []

    for item in tqdm(data, desc=f"  Encoding (Qwen/{modality})"):
        if modality == "audio":
            user_content = [{"type": "audio", "audio": item[src_audio_field]}]
        elif modality == "text":
            user_content = [{"type": "text", "text": f"Source: {item['src']}"}]
        else:  # audiotext: audio first, source text follows
            user_content = [
                {"type": "audio", "audio": item[src_audio_field]},
                {"type": "text",  "text": f"Source: {item['src']}"},
            ]

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": user_content},
        ]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=not last_user_token, tokenize=False
        )
        if probe_prompt is not None:
            text = text + probe_prompt

        if not needs_audio:
            inputs = processor(text=[text], return_tensors="pt", padding=True)
        elif process_mm_info is not None:
            audios, images, videos = process_mm_info([conversation], use_audio_in_video=False)
            inputs = processor(
                text=[text], audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=False,
            )
        else:
            wav_path = item[src_audio_field]
            audio_data, sr = sf.read(wav_path)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            if sr != 16000:
                import torchaudio
                waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                audio_data = waveform.squeeze(0).numpy()
                sr = 16000
            inputs = processor(
                text=[text], audio=[audio_data], sampling_rate=sr,
                return_tensors="pt", padding=True,
            )

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.thinker(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        input_ids = inputs["input_ids"][0]

        if last_user_token or probe_prompt is not None:
            # last_user_token: sequence ends at user turn (no generation prompt added),
            # so the last real token is the last token of the user input.
            # probe_prompt: sequence ends at the probe string in the assistant turn.
            # In both cases, extract at the last non-padding token.
            attn_mask = inputs["attention_mask"][0]
            extract_pos = attn_mask.nonzero(as_tuple=True)[0][-1].item()
        else:
            positions = (input_ids == _QWEN_ASSISTANT_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(positions) == 0:
                raise ValueError(
                    f"'assistant' token (id={_QWEN_ASSISTANT_TOKEN_ID}) not found. "
                    f"Check that the chat template is correct."
                )
            extract_pos = positions[-1].item()

        hidden = outputs.hidden_states[layer_idx]
        emb = hidden[0, extract_pos, :].float().cpu()
        embeddings.append(emb)

    return torch.stack(embeddings)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Probe COMET sentence embeddings with MLP classifiers.",
    )
    parser.add_argument("--train-data", required=True, help="Path to training JSONL file.")
    parser.add_argument("--dev-data", required=True, help="Path to validation JSONL file.")
    parser.add_argument(
        "--attributes", nargs="+", required=True,
        help="One or more JSONL column names to use as probe labels (e.g. --attributes lp domain).",
    )
    parser.add_argument(
        "--model", default="Unbabel/wmt22-cometkiwi-da",
        help="COMET model name, HuggingFace repo ID, or local checkpoint path.",
    )
    parser.add_argument(
        "--model-type", choices=["comet", "speechcomet", "qwen"], default="comet",
        help="Model framework. 'qwen' uses Qwen2.5-Omni and extracts the LLM hidden "
             "state at the assistant token as the audio representation.",
    )
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding extraction batch size.")
    parser.add_argument("--probe-batch-size", type=int, default=256,
                        help="MLP training batch size.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Probe training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Probe learning rate.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256],
                        help="MLP hidden layer sizes.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="MLP dropout rate.")
    parser.add_argument("--src-lang", default=None,
                        help="Filter to only keep entries with this src_lang (e.g. 'en').")
    parser.add_argument("--train-frac", type=float, default=1.0,
                        help="Fraction of training data to use (0, 1]. Default: 1.0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto",
                        help="Device: cuda, cpu, or auto.")
    parser.add_argument("--embeddings-dir", default=None,
                        help="Cache directory for extracted embeddings.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use half precision for the COMET model.")
    parser.add_argument("--src-audio-field", default="src_audio",
                        help="JSONL field name containing audio paths for speechCOMET/Qwen audio models. "
                             "Defaults to 'src_audio'.")
    parser.add_argument("--qwen-layer", type=int, default=-1,
                        help="LLM layer index to read the assistant-token hidden state from "
                             "(0 = embedding layer, -1 = last layer). Only used with --model-type qwen.")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.",
                        help="System prompt passed to Qwen2.5-Omni. Keep neutral to avoid "
                             "text-side confounds. Only used with --model-type qwen.")
    parser.add_argument("--probe-prompt", default=None,
                        help="If set, appended after the generation-prompt suffix "
                             "(<|im_start|>assistant\\n<probe-prompt>) and the last real "
                             "token of the resulting input is used as the utterance "
                             "representation instead of the bare 'assistant' token. "
                             "E.g. --probe-prompt \"The speaker is\". "
                             "Only used with --model-type qwen.")
    parser.add_argument("--qwen-adapter", default=None,
                        help="HuggingFace repo ID or local path to a PEFT adapter to apply "
                             "on top of the Qwen2.5-Omni model (e.g. maikezu/main-SpeechLLM-Speech-FT). "
                             "Weights are merged before probing. Only used with --model-type qwen.")
    parser.add_argument("--qwen-modality", choices=["audio", "text", "audiotext"], default="audio",
                        help="Input modality for Qwen: 'audio' (wav only, default), "
                             "'text' (src text only), or 'audiotext' (wav followed by src text). "
                             "Only used with --model-type qwen.")
    parser.add_argument("--qwen-last-user-token", action="store_true",
                        help="Extract the hidden state at the last token of the user turn "
                             "instead of the assistant/probe-prompt token. Uses "
                             "add_generation_prompt=False so the sequence ends at the user turn. "
                             "Pair with --system-prompt to set task context. "
                             "Only used with --model-type qwen.")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Determine whether audio data is needed before loading the model so we
    # can validate the JSONL fields early.  We detect modality from the model
    # after loading it, but we can do a best-effort check: if --model-type is
    # speechcomet we defer the audio field check until after model load.
    # For now load data without the audio requirement and re-validate below.
    print(f"Loading train data from {args.train_data} ...")
    train_data = load_jsonl(args.train_data, args.attributes)
    if args.src_lang:
        train_data = [d for d in train_data if d.get("src_lang") == args.src_lang]
        print(f"  Filtered to src_lang='{args.src_lang}': {len(train_data)} train samples.")
    if args.train_frac < 1.0:
        n = max(1, int(len(train_data) * args.train_frac))
        rng = random.Random(args.seed)
        train_data = rng.sample(train_data, n)
    print(f"  {len(train_data)} train samples loaded.")
    print(f"Loading dev data from {args.dev_data} ...")
    dev_data = load_jsonl(args.dev_data, args.attributes)
    if args.src_lang:
        dev_data = [d for d in dev_data if d.get("src_lang") == args.src_lang]
        print(f"  Filtered to src_lang='{args.src_lang}': {len(dev_data)} dev samples.")
    print(f"  {len(dev_data)} dev samples loaded.")
    print(f"  Attribute columns: {args.attributes}")

    # -------------------------------------------------------------------------
    # Qwen2.5-Omni path: load model, extract assistant-token embeddings, probe
    # -------------------------------------------------------------------------
    if args.model_type == "qwen":
        qwen_model_name = args.model if args.model != "Unbabel/wmt22-cometkiwi-da" else "Qwen/Qwen2.5-Omni-7B"

        # Build a modality tag that captures all Qwen-specific extraction parameters
        # so different layer/modality/prompt/adapter combos never share a cache file.
        _qwen_tag = f"qwen_layer{args.qwen_layer}_{args.qwen_modality}"
        if args.qwen_last_user_token:
            _qwen_tag += "_lastuser"
        if args.system_prompt != "You are a helpful assistant.":
            _qwen_tag += "_sys" + hashlib.md5(args.system_prompt.encode()).hexdigest()[:8]
        if args.probe_prompt is not None:
            _qwen_tag += "_prompt" + hashlib.md5(args.probe_prompt.encode()).hexdigest()[:8]
        if args.qwen_adapter is not None:
            _qwen_tag += "_adapter" + hashlib.md5(args.qwen_adapter.encode()).hexdigest()[:8]

        train_cache_path = None
        dev_cache_path = None
        if args.embeddings_dir:
            os.makedirs(args.embeddings_dir, exist_ok=True)
            # When train_frac < 1 each seed samples a different subset, so the
            # train cache must be keyed on the seed to avoid embedding/label mismatch.
            train_tag = _qwen_tag + (f"_seed{args.seed}" if args.train_frac < 1.0 else "")
            train_cache_path = os.path.join(
                args.embeddings_dir,
                f"{_cache_key(args.train_data, qwen_model_name, train_tag)}.pt",
            )
            dev_cache_path = os.path.join(
                args.embeddings_dir,
                f"{_cache_key(args.dev_data, qwen_model_name, _qwen_tag)}.pt",
            )

        _train_cached = train_cache_path and os.path.exists(train_cache_path)
        _dev_cached = dev_cache_path and os.path.exists(dev_cache_path)

        if _train_cached and _dev_cached:
            print(f"Loading cached Qwen train embeddings from {train_cache_path}")
            train_src_emb = torch.load(train_cache_path, weights_only=True)
            print(f"Loading cached Qwen dev embeddings from {dev_cache_path}")
            dev_src_emb = torch.load(dev_cache_path, weights_only=True)
        else:
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

            print(f"Loading Qwen2.5-Omni model: {qwen_model_name} ...")
            qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                qwen_model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
            if args.qwen_adapter is not None:
                from peft import PeftModel
                print(f"  Applying adapter: {args.qwen_adapter} ...")
                qwen_model.thinker = PeftModel.from_pretrained(qwen_model.thinker, args.qwen_adapter)
                qwen_model.thinker = qwen_model.thinker.merge_and_unload()
                print("  Adapter merged.")
            qwen_model.eval()
            qwen_processor = Qwen2_5OmniProcessor.from_pretrained(qwen_model_name)
            print(f"  Loaded. Layer index for probing: {args.qwen_layer}")
            print(f"  System prompt: {args.system_prompt!r}")
            print(f"  Probe prompt:  {args.probe_prompt!r} "
                  f"({'last real token' if args.probe_prompt is not None else 'assistant token'})")

            if not _train_cached:
                print("\nExtracting train embeddings (Qwen)...")
                train_src_emb = extract_qwen_embeddings(
                    qwen_model, qwen_processor, train_data,
                    args.src_audio_field, args.system_prompt, args.qwen_layer, args.probe_prompt,
                    modality=args.qwen_modality, last_user_token=args.qwen_last_user_token,
                )
                if train_cache_path:
                    torch.save(train_src_emb, train_cache_path)
                    print(f"Cached Qwen train embeddings to {train_cache_path}")
            else:
                print(f"Loading cached Qwen train embeddings from {train_cache_path}")
                train_src_emb = torch.load(train_cache_path, weights_only=True)

            if not _dev_cached:
                print("\nExtracting dev embeddings (Qwen)...")
                dev_src_emb = extract_qwen_embeddings(
                    qwen_model, qwen_processor, dev_data,
                    args.src_audio_field, args.system_prompt, args.qwen_layer, args.probe_prompt,
                    modality=args.qwen_modality, last_user_token=args.qwen_last_user_token,
                )
                if dev_cache_path:
                    torch.save(dev_src_emb, dev_cache_path)
                    print(f"Cached Qwen dev embeddings to {dev_cache_path}")
            else:
                print(f"Loading cached Qwen dev embeddings from {dev_cache_path}")
                dev_src_emb = torch.load(dev_cache_path, weights_only=True)

            del qwen_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("\nQwen model released from memory.\n")

        run_attribute_probes(
            args.attributes, train_data, dev_data,
            {"src_emb": train_src_emb}, {"src_emb": dev_src_emb}, args,
        )
        return

    # -------------------------------------------------------------------------
    # Load model
    print(f"Loading {args.model_type} model: {args.model} ...")
    if args.model_type == "speechcomet":
        try:
            from speechcomet.models import load_from_checkpoint as sc_load
        except ImportError:
            raise ImportError(
                "speechcomet is not installed. "
                "Install it from https://github.com/zouharvi/speechCOMET"
            )
        if os.path.isfile(args.model):
            checkpoint_path = args.model
        else:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                repo_id=args.model, filename="checkpoints/model.ckpt"
            )
            # speechCOMET requires hparams.yaml in the same folder as the checkpoint
            hparams_path = hf_hub_download(
                repo_id=args.model, filename="hparams.yaml"
            )
            ckpt_dir = os.path.dirname(checkpoint_path)
            hparams_dest = os.path.join(ckpt_dir, "hparams.yaml")
            if not os.path.exists(hparams_dest):
                import shutil
                shutil.copy(hparams_path, hparams_dest)
        model = sc_load(checkpoint_path)
    else:
        if os.path.isfile(args.model):
            checkpoint_path = args.model
        else:
            checkpoint_path = download_model(args.model)
        model = _comet_load_from_checkpoint(checkpoint_path, local_files_only=True)
    model.eval()
    if args.fp16:
        model.half()
    model.to(args.device)

    # Detect source modality for speechCOMET models
    modality = getattr(model, "input_modality", "text")
    print(f"  Model loaded on {args.device} "
          f"(encoder output_units={model.encoder.output_units}, "
          f"src modality={modality})")

    # Validate that audio field is present when needed
    if modality in ("audio", "audiotext"):
        for split_name, split_data in [("train", train_data), ("dev", dev_data)]:
            missing = [i for i, d in enumerate(split_data) if args.src_audio_field not in d]
            if missing:
                raise ValueError(
                    f"{split_name} data: {len(missing)} rows missing field "
                    f"'{args.src_audio_field}' (required for modality='{modality}'). "
                    f"First missing at row {missing[0]}."
                )

    # Extract embeddings for both splits
    print("\nExtracting train embeddings...")
    train_emb_data = extract_all_embeddings(model, train_data, args.train_data, args, modality)
    print("\nExtracting dev embeddings...")
    dev_emb_data = extract_all_embeddings(model, dev_data, args.dev_data, args, modality)

    # Free COMET model to save GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nCOMET model released from memory.\n")

    # Train probes for each (attribute, embedding) combination
    run_attribute_probes(args.attributes, train_data, dev_data, train_emb_data, dev_emb_data, args)


if __name__ == "__main__":
    main()
