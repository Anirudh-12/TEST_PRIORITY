#!/bin/bash
set -e

# Fixed Test Set (do not re-mine, already completed)
# fastapi
# sanic

# Train Set (mine with a cap)
PROJECTS=("httpie" "thefuck" "PySnooper" "ansible" "cookiecutter" "keras" "luigi" "matplotlib" "pandas" "scrapy" "spacy" "tornado" "youtube-dl")

MAX_BUGS=10

echo "Starting massive BugsInPy scale-up..."

for PROJECT in "${PROJECTS[@]}"; do
    echo "============================================================"
    echo "Mining project: $PROJECT (Max $MAX_BUGS bugs)"
    echo "============================================================"
    
    # We allow failures here so it doesn't crash the whole loop if one project fails
    uv run python scripts/collect_dataset.py --only "$PROJECT" --with-coverage --max-bugs $MAX_BUGS || echo "Project $PROJECT had some errors, continuing..."
done

echo "Scale-up mining complete!"
