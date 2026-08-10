Write-Host "Cleaning old results..."
if (Test-Path "data/results") {
    Remove-Item -Recurse -Force "data/results"
}
New-Item -ItemType Directory -Force "data/results" | Out-Null

Write-Host "Running baselines..."
uv run python -m experiments.run_experiments

Write-Host "Running LOPO RL..."
uv run python -m experiments.run_lopo_rl

Write-Host "Running statistical validation..."
uv run python evaluation/statistical.py | Out-File -FilePath "data/results/statistical_report.txt" -Encoding utf8
Get-Content "data/results/statistical_report.txt"

Write-Host "Plotting..."
uv run python analysis/plot_results.py
Write-Host "Done!"
