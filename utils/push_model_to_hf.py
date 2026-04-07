import speechcomet
import glob
import os
import shutil
import tempfile
from huggingface_hub import HfApi
from tqdm import tqdm

api = HfApi()

TRAINED_MODELS_DIR = "trained_models"

# Maps HF repo name -> local model folder to push (highest-epoch version)
models = {
    "harris":                    "harris-20ep-continue",
    "lewis":                     "lewis-10ep",
    "mull-attn":                 "mull-attn-10ep",
    "mull-attn-lora":            "mull-attn-lora-10ep",
    "mull-avg":                  "mull-avg-20ep",
    "mull-avg-lora":             "mull-avg-lora-10ep",
    "orkney-avg":                "orkney-avg-20ep",
    "orkney-concat":             "orkney-concat-20ep",
    "orkney-sum":                "orkney-sum-20ep",
    "orkney-sum-from-text-ckpt": "orkney-sum-from-text-ckpt-20ep",
    "shetland":                  "shetland-20ep",
    "skye":                      "skye-20ep",
}
models = {hf_name: os.path.join(TRAINED_MODELS_DIR, local) for hf_name, local in models.items()}

for model_name, model_path in tqdm(models.items(), desc="Uploading models"):
    ckpt_dir = os.path.join(model_path, "checkpoints")
    matches = glob.glob(os.path.join(ckpt_dir, "epoch=*-*.ckpt"))
    last_checkpoint = max(
        matches,
        key=lambda p: int(os.path.basename(p).split("epoch=")[1].split("-")[0])
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
