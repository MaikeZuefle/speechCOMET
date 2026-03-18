import speechcomet
import glob
import os
import shutil
import tempfile
from huggingface_hub import HfApi
from tqdm import tqdm

api = HfApi()

TRAINED_MODELS_DIR = "trained_models"

models = {
    name: os.path.join(TRAINED_MODELS_DIR, name)
    for name in sorted(os.listdir(TRAINED_MODELS_DIR))
    if glob.glob(os.path.join(TRAINED_MODELS_DIR, name, "checkpoints", "epoch=*-*.ckpt"))
}

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

        repo_id = f"maikezu/{model_name}"
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            repo_type="model",
        )

    print(f"Uploaded {model_name} -> https://huggingface.co/{repo_id}")
