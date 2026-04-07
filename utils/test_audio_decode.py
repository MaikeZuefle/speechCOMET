"""Quick test: verify torchaudio decode path in sonar.py works on a few examples.

Works on login nodes (no FFmpeg/torchcodec needed) by loading with decode=False
and then exercising the same decode logic as sonar.py's prepare_sample.
"""
import io
import torchaudio
from datasets import load_dataset, Audio

# Load raw bytes (decode=False avoids torchcodec entirely for loading)
ds = load_dataset("maikezu/ben_nevis_tts", split="train[:10]")
ds = ds.cast_column("audio", Audio(decode=False))

for i, row in enumerate(ds):
    raw = row["audio"]  # {"bytes": ..., "path": ...}

    # Simulate what sonar.py's prepare_sample does for an AudioDecoder
    # with _hf_encoded = {"bytes": ..., "path": ...}
    if raw.get("bytes"):
        wf, sr = torchaudio.load(io.BytesIO(raw["bytes"]))
        via = "torchaudio (bytes)"
    elif raw.get("path"):
        wf, sr = torchaudio.load(raw["path"])
        via = "torchaudio (path)"
    else:
        raise ValueError(f"No bytes or path for example {i}")

    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)

    print(f"[{i}] via={via}  shape={wf.shape}  sr={sr}  dur={wf.shape[-1]/sr:.2f}s")

print("\nAll 10 examples decoded successfully.")
