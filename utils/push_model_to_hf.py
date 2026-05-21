import speechcomet
import glob
import os
import re
import shutil
import tempfile
from huggingface_hub import HfApi
from tqdm import tqdm

api = HfApi()

TRAINED_MODELS_DIR = "trained_models"

# Maps HF repo name -> local model folder to push (highest-epoch version)
models = {
    # "harris":                         "harris-20ep",
    # "lewis":                          "lewis-10ep",
    # "mull-attn":                      "mull-attn-10ep",
    # "mull-attn-lora":                 "mull-attn-lora-10ep",
    # "mull-avg":                       "mull-avg-20ep",
    # "mull-avg-lora":                  "mull-avg-lora-10ep",
    # "orkney-avg":                     "orkney-avg-20ep",
    # "orkney-concat":                  "orkney-concat-20ep",
    # "orkney-sum":                     "orkney-sum-20ep",
    # "orkney-sum-from-text-ckpt":      "orkney-sum-from-text-ckpt-20ep",
    # "shetland":                       "shetland-20ep",
    # "skye":                           "skye-20ep",
    # TTS-augmented models
    # "Lewis-TTS":                      "bute-pretrain",
    # "Lewis-TTS-unfreeze":             "bute-pretrain-unfreeze-sonar",
    # "Lewis-TTS-Harris":               "bute-train",
    # "Lewis-TTS-unfreeze-Harris":      "bute-pretrain-unfreeze-sonar-train-freeze",
    # "Lewis-TTS-unfreeze-Harris-unfreeze": "bute-pretrain-unfreeze-sonar-train-unfreeze",
    # # Frozen-encoder ablation models
    "frozen-ablation-SpeechCOMET-SONAR":   "harris-FT-sonar",
    "frozen-ablation-SpeechCOMET-Whisper": "mull-attn-10ep",
    "frozen-ablation-SpeechCOMET-textaudio":   "orkney-sum-20ep",
    # Main table models (paper names)
    "main-COMETKiwi-RoBERTa-WMT":    "lewis-10ep",
    "main-COMETKiwi-RoBERTa-IWSLT":  "skye-20ep",
    "main-SpeechCOMET-SONAR":        "harris-FT-sonar",
    "main-SpeechCOMET-Whisper":      "mull-attn-from-ckpt",
    "main-SpeechCOMET-textaudio":              "orkney-sum-from-text-ckpt-20ep",
    "main-SpeechCOMET-textaudio-large":        "orkney-sum-from-text-ckpt-BIG",
}
models = {hf_name: os.path.join(TRAINED_MODELS_DIR, local) for hf_name, local in models.items()}

for model_name, model_path in tqdm(models.items(), desc="Uploading models"):
    ckpt_dir = os.path.join(model_path, "checkpoints")
    matches = [
        p for p in glob.glob(os.path.join(ckpt_dir, "epoch=*-val_kendall=*.ckpt"))
        if not os.path.basename(p).startswith("worse_")
        and re.search(r"val_kendall=(\d+\.\d+)", os.path.basename(p))
    ]
    last_checkpoint = max(
        matches,
        key=lambda p: float(re.search(r"val_kendall=(\d+\.\d+)", os.path.basename(p)).group(1))
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy entire model folder except the checkpoints dir
        for item in os.listdir(model_path):
            if item == "checkpoints":
                continue
            src = os.path.join(model_path, item)
            dst = os.path.join(tmp_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        # Copy only the last checkpoint
        tmp_ckpt_dir = os.path.join(tmp_dir, "checkpoints")
        os.makedirs(tmp_ckpt_dir)
        shutil.copy2(last_checkpoint, os.path.join(tmp_ckpt_dir, "model.ckpt"))  # rename here

        repo_id = f"maikezu/{model_name}"  # model_name is the HF repo name
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            repo_type="model",
        )

    print(f"Uploaded {model_name} -> https://huggingface.co/{repo_id}")

# ── LoRA adapters (SpeechLLM +FT) ─────────────────────────────────────────────
LORA_BASE = "speechllm-baselines/saves/qwen2.5-omni-7b/lora"

lora_models = {
    "main-SpeechLLM-Speech-FT":  "audio",
    "main-SpeechLLM-Text-FT":    "text",
    "main-SpeechLLM-SpTxt-FT":   "textaudio",
}

for model_name, subdir in tqdm(lora_models.items(), desc="Uploading LoRA adapters"):
    adapter_path = os.path.join(LORA_BASE, subdir)
    repo_id = f"maikezu/{model_name}"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*", "README.md"],  # skip intermediate checkpoints and invalid README
    )

    print(f"Uploaded {model_name} -> https://huggingface.co/{repo_id}")
