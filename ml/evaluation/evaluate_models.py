import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--dataset", required=True)
args = parser.parse_args()

if not Path(args.model).exists() or not Path(args.dataset).exists():
    raise SystemExit("Model and dataset paths are required for evaluation. No metrics were generated.")

raise SystemExit("Evaluation scaffold ready for project-specific metrics implementation.")
