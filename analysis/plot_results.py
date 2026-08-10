import sqlite3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DB_PATH = RESULTS_DIR / "results_db.sqlite"


def load_data(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)
        
    conn = sqlite3.connect(db_path)
    # Query data and average over random seeds
    query = """
        SELECT
            policy_name,
            budget_fraction,
            AVG(fault_detected) as fdr,
            AVG(apfdc) as apfdc
        FROM episodes
        GROUP BY policy_name, budget_fraction
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Clean up policy names (remove random_seedX from name to group them as "random")
    df['policy_name'] = df['policy_name'].apply(lambda x: 'random' if x.startswith('random_seed') else x)
    
    # Try loading LOPO DB if it exists
    lopo_db_path = db_path.parent / "lopo_results_db.sqlite"
    if lopo_db_path.exists():
        conn_lopo = sqlite3.connect(lopo_db_path)
        df_lopo = pd.read_sql_query(query, conn_lopo)
        conn_lopo.close()
        df = pd.concat([df, df_lopo], ignore_index=True)

    # Try loading Static Split DB if it exists
    static_db_path = db_path.parent / "static_results_db.sqlite"
    if static_db_path.exists():
        conn_static = sqlite3.connect(static_db_path)
        df_static = pd.read_sql_query(query, conn_static)
        conn_static.close()
        df = pd.concat([df, df_static], ignore_index=True)
    
    # Re-aggregate to average the random seeds properly
    df = df.groupby(['policy_name', 'budget_fraction']).mean().reset_index()
    
    return df


def plot_metrics(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Plot FDR vs Budget
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="budget_fraction",
        y="fdr",
        hue="policy_name",
        marker="o",
        linewidth=2,
        markersize=8
    )
    plt.title("Fault Detection Rate (FDR) vs. Budget Constraint", fontsize=14)
    plt.xlabel("Budget Fraction (% of Total Suite Time)", fontsize=12)
    plt.ylabel("FDR (Probability of Finding Bug)", fontsize=12)
    plt.xticks([0.05, 0.10, 0.25, 0.50, 1.00], ['5%', '10%', '25%', '50%', '100%'])
    plt.ylim(0, 1.05)
    plt.legend(title="Policy")
    plt.tight_layout()
    fdr_path = output_dir / "fdr_vs_budget.png"
    plt.savefig(fdr_path, dpi=300)
    print(f"Saved FDR plot to {fdr_path}")
    plt.close()
    
    # 2. Plot APFDc vs Budget
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="budget_fraction",
        y="apfdc",
        hue="policy_name",
        marker="s",
        linewidth=2,
        markersize=8
    )
    plt.title("APFDc (Cost-aware Fault Detection) vs. Budget Constraint", fontsize=14)
    plt.xlabel("Budget Fraction (% of Total Suite Time)", fontsize=12)
    plt.ylabel("APFDc Score (Higher = Faster Detection)", fontsize=12)
    plt.xticks([0.05, 0.10, 0.25, 0.50, 1.00], ['5%', '10%', '25%', '50%', '100%'])
    plt.ylim(0, 1.05)
    plt.legend(title="Policy")
    plt.tight_layout()
    apfdc_path = output_dir / "apfdc_vs_budget.png"
    plt.savefig(apfdc_path, dpi=300)
    print(f"Saved APFDc plot to {apfdc_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results from SQLite database.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to results SQLite DB.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR, help="Directory to save plots.")
    args = parser.parse_args()
    
    df = load_data(args.db)
    print("Loaded aggregated results from database.")
    
    plot_metrics(df, args.output)


if __name__ == "__main__":
    main()
