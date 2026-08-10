# Budget-Constrained Online Regression Test Selection

> **Research Question**: Under a strict execution budget, does a lightweight online test-selection policy provide a statistically meaningful advantage over random, coverage-based, historical, static-ML, and existing adaptive test-selection baselines when evaluated on previously unseen Python projects from BugsInPy?

## Quick Start

### Prerequisites
- Windows with WSL 2 (Ubuntu) installed
- uv (`winget install astral-sh.uv`)

### 1. Set up the Windows Python environment
```bash
uv sync
```

### 2. Set up the WSL environment (run once inside Ubuntu WSL)
```bash
wsl -d Ubuntu
bash /mnt/c/Users/aksha/OneDrive/Documents/TEST_PRIORITY/scripts/setup_wsl_environment.sh
```

### 3. Process a single bug (Phase 1 validation)
```bash
uv run python -m bugsinpy.process --project thefuck --bug 1 --skip-coverage
```

Output: `data/processed/thefuck/1.json`

### 4. Run leakage checks
```bash
uv run pytest tests/test_leakage.py -v
```

### 5. Run a full experiment (after processing 5+ bugs)
```bash
uv run python experiments/run.py \
    --dataset bugsinpy \
    --experiment cross_project \
    --policy random \
    --budget 25 \
    --seed 42
```

---

## Project Structure

```
TEST_PRIORITY/
├── bugsinpy/           # BugsInPy interface (checkout, runner, coverage, metadata)
├── analysis/           # Diff analysis, AST, features
├── environment/        # Agent state, budget, simulator
├── policies/           # Baselines + adaptive policy
├── training/           # ML model training
├── evaluation/         # Metrics, statistical tests, leakage checks
├── experiments/        # Experiment runner + configs
├── data/
│   ├── raw/            # BugsInPy clone
│   ├── processed/      # Experiment records (<project>/<bug_id>.json)
│   ├── schemas/        # JSON Schema for validation
│   └── dataset_status.jsonl   # Bug processing status log
├── docs/
│   └── literature_matrix.csv
├── tests/              # Unit + leakage tests
└── scripts/
    └── setup_wsl_environment.sh
```

---

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | 🔄 In Progress | BugsInPy environment setup in WSL |
| 2 | ⬜ | Single bug end-to-end pipeline |
| 3 | ⬜ | Schema + validation |
| 4 | ⬜ | Scale to 5–10 bugs |
| 5 | ⬜ | Random + Coverage baselines |
| 6 | ⬜ | Static ML baseline |
| 7 | ⬜ | Sequential simulator |
| 8 | ⬜ | Adaptive policy |
| 9 | ⬜ | Cross-project experiments |
| 10 | ⬜ | Ablations + scaling |

---

## Data Leakage Prevention

This is a research project. Leakage is a critical failure.

Every test record has three clearly separated sections:

```json
{
  "PRE_EPISODE_FEATURES": { ... },   ← Policy MAY see this
  "GROUND_TRUTH_LABELS":  { ... }    ← Policy MUST NOT see this
}
```

Run `pytest tests/test_leakage.py` before any experiment.

---

## BugsInPy Projects Available

| Project | Bugs | Python |
|---------|------|--------|
| thefuck | 32 | 3.7–3.8 |
| pandas | 150+ | 3.8 |
| keras | 50+ | 3.6–3.8 |
| scrapy | 40+ | 3.6–3.8 |
| black | 23 | 3.6–3.8 |
| cookiecutter | 4 | 3.7 |
| fastapi | 16 | 3.8 |
| matplotlib | 30+ | 3.8 |
| tornado | 16 | 3.8 |
| tqdm | 9 | 3.7 |
| httpie | 5 | 3.8 |
| luigi | 33 | 3.8 |
| sanic | 5 | 3.8 |
| spacy | 10 | 3.6 |
| ansible | 18 | 3.8 |
| youtube-dl | 25 | 3.8 |
| PySnooper | 3 | 3.6 |

**Start with**: `thefuck` (small, fast tests, well-structured) or `cookiecutter` (few bugs, simple environment).

---

## Evaluation Metrics

- **FDR** — Fault Detection Rate
- **TTD** — Tests to Detection  
- **T2D** — Time to Detection
- **TSR** — Test Suite Reduction (`1 - selected/total`)
- **APFD** — Average Percentage of Faults Detected
- **APFDc** — Cost-weighted APFD
- **BSR** — Budget Success Rate at 5%, 10%, 25%, 50%

---

## Hardware

- CPU: Intel i5-13420H
- RAM: 24 GB
- GPU: None required
- Python: 3.11+ (orchestration), 3.6–3.8 (bug environments via pyenv/WSL)
