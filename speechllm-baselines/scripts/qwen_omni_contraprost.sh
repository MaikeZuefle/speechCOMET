cd "$(dirname "$0")/../src" || exit 1

python generate_qwen_omni.py \
    --contraprost-dir ../../data/contraProST
