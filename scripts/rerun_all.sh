#!/bin/bash
set -e
echo "Cleaning old results..."
rm -rf data/results
mkdir -p data/results

echo "Running baselines..."
uv run python -m experiments.run_experiments

echo "Running LOPO RL..."
uv run python -m experiments.run_lopo_rl

echo "Running statistical validation..."
uv run python evaluation/statistical.py > data/results/statistical_report.txt
cat data/results/statistical_report.txt

echo "Plotting..."
uv run python analysis/plot_results.py
echo "Done!"
