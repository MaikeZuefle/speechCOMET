import asr_comet
import asr_comet_partial
import argparse
import json
import os

methods = {
    "asr_comet": asr_comet,
    "asr_comet_partial": asr_comet_partial,
}

args = argparse.ArgumentParser()
args.add_argument("-m", "--method", type=str, required=True, choices=methods.keys())
args.add_argument("-i", "--input", type=str, required=True)
args.add_argument("-o", "--output", type=str, required=True)
parsed = args.parse_args()

os.makedirs(os.path.dirname(parsed.output), exist_ok=True)

with open(parsed.input, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

data_out = methods[parsed.method].run(data)

with open(parsed.output, "w") as f:
    for score in data_out:
        f.write(f"{score}\n")


"""
sbatch_gpu_short wmt25_comet "python3 baselines -i data/wmt25.jsonl -m asr_comet -o data/output/wmt25_comet.jsonl"
sbatch_gpu_short wmt25_comet_partial "python3 baselines -i data/wmt25.jsonl -m asr_comet_partial -o data/output/wmt25_comet_partial.jsonl"
"""