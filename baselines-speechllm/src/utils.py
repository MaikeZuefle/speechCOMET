"""Shared data-loading utilities for speechllm baselines."""
import os
import sys

# Use shared loaders and evaluation functions from speechcomet-eval/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "speechcomet-eval"))
from mustshe_eval import load_csv_files as load_mustshe_csv_files
from mustshe_eval import compute_results as compute_mustshe_results
from mustshe_eval import print_mustshe_pivot
from eval_utils import load_contraprost_csv_files
from contraprost_eval import compute_results as compute_contraprost_results
from contraprost_eval import print_contraprost_results
