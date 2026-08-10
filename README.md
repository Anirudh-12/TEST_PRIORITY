# Language-Agnostic Reinforcement Learning for Test Prioritization

This repository contains the full end-to-end pipeline for training and evaluating a Contextual Bandit (Neural Network) Reinforcement Learning agent for Test Case Prioritization (TCP). 

The framework is explicitly designed to be **language-agnostic**—while it currently mines Python repositories via `BugsInPy`, all extracted features (coverage density, historical failure rate, normalized runtimes, etc.) are mapped to a continuous $[0, 1]$ vector space. This ensures the same Neural Bandit architecture can seamlessly evaluate Java datasets (e.g., `Defects4J`) or other languages in the future without modification.

## 🚀 Key Features

* **Fully Automated Data Collection**: Scripts to automatically checkout historical buggy commits, build virtual environments, and extract line-level execution traces and performance metrics.
* **Continuous Feature Space**: A universal, normalized input space that prevents the RL model from overfitting to language-specific attributes.
* **Leave-One-Project-Out (LOPO) Cross-Validation**: Rigorous training loop that evaluates the RL agent exclusively on entirely unseen projects, proving true generalization.
* **Statistical Validation**: Built-in scripts to compute Wilcoxon Signed-Rank tests and Vargha-Delaney $\hat{A}_{12}$ effect sizes against heuristic baselines.

## 📊 Experimental Results (37 Bugs)

We evaluated the `neural_bandit` agent against 4 standard heuristic baselines (`random`, `coverage_greedy`, `historical_failure_rate`, `cost_weighted_coverage`) on 37 bugs across 4 major Python projects (`fastapi`, `sanic`, `thefuck`, `httpie`).

The results revealed that at highly constrained test budgets, the Reinforcement Learning agent **statistically significantly outperforms** historical-based heuristics:

| Budget Constraint | RL vs Historical Baseline (p-value) | Winner |
|-------------------|-------------------------------------|--------|
| **5%**            | 0.0007                              | RL Agent |
| **10%**           | 0.0007                              | RL Agent |
| **25%**           | 0.0045                              | RL Agent |

*At higher budgets (50%+), the greedy coverage policy ultimately catches up and dominates as it has sufficient time to solve the Set Cover problem perfectly.*

### Visualizations

The generated comparative charts for Fault Detection Rate (FDR) and Cost-aware Average Percentage of Faults Detected (APFDc) are saved in:
`data/results/plots/`

## 🛠️ Usage

### 1. Prerequisites
- Python 3.10+
- `uv` (Fast Python Package Installer)
- WSL (Ubuntu) is highly recommended for the BugsInPy data collection step due to legacy python compilation requirements.

Install dependencies:
```bash
uv pip install -r requirements.txt
```

### 2. Mining Data
To collect coverage matrices and execution traces from a historical bug (e.g., FastAPI Bug #12):
```bash
uv run python scripts/collect_dataset.py --only fastapi --with-coverage
```

### 3. Evaluating Baselines & Training the RL Agent
You can run the entire evaluation suite (baselines + LOPO RL + Statistics + Plots) using the provided automation scripts natively on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rerun_all.ps1
```
Or on Linux/WSL:
```bash
bash scripts/rerun_all.sh
```

## 🏗️ Repository Structure

* `bugsinpy/`: The ingestion engine. Automates repo cloning, virtual env construction, and Pytest coverage matrix generation.
* `data/`: Contains raw logs, processed JSON records (`data/processed`), and SQLite result databases (`data/results`).
* `policies/`: Contains the base abstraction and the `NeuralBanditPolicy` PyTorch implementation.
* `experiments/`: The runner scripts for heuristic baselines (`run_experiments.py`) and cross-validation training (`run_lopo_rl.py`).
* `evaluation/`: Contains `statistical.py` for computing non-parametric statistical significance tests.
* `analysis/`: Contains `plot_results.py` for generating `seaborn` line charts.
