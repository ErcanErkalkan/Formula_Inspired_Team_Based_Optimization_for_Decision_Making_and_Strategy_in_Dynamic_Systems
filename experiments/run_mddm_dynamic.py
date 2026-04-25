import os
import sys
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make sure we can import from experiments
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from run_dynamic_asoc_suite import (
    PROTOCOLS, PROBLEMS, SEEDS, FIXED_BUDGET_TARGET, 
    calibrate_budget_pop_sizes, run_task, RESULTS_DIR, average_ranks, summarize_metric, pairwise_tests, summarize_eval_budget
)

def run_mddm_append():
    csv_path = RESULTS_DIR / "asoc_dynamic_raw_metrics.csv"
    if not csv_path.exists():
        print("Raw metrics CSV not found!")
        return

    df_existing = pd.read_csv(csv_path)
    
    fixed_budget_pop_sizes = calibrate_budget_pop_sizes()
    
    tasks = []
    
    for algo in ["MDDM-DMOEA", "PPS-DMOEA"]:
        if algo not in df_existing["algorithm"].values:
            # Generation matched
            for protocol in PROTOCOLS.keys():
                for problem in PROBLEMS:
                    for seed in SEEDS:
                        tasks.append({
                            "family": "main",
                            "protocol": protocol,
                            "problem": problem,
                            "algorithm": algo,
                            "seed": seed,
                            "pop_size": PROTOCOLS[protocol]["pop_size"],
                            "generations": PROTOCOLS[protocol]["generations"]
                        })
                        
            # Budget matched
            for protocol in PROTOCOLS.keys():
                for problem in PROBLEMS:
                    for seed in SEEDS:
                        pop_sz = fixed_budget_pop_sizes[protocol].get(algo, PROTOCOLS[protocol]["pop_size"])
                        tasks.append({
                            "family": "budget",
                            "protocol": protocol,
                            "problem": problem,
                            "algorithm": algo,
                            "seed": seed,
                            "pop_size": pop_sz,
                            "generations": PROTOCOLS[protocol]["generations"]
                        })

    if not tasks:
        print("Both MDDM-DMOEA and PPS-DMOEA already exist in dynamic results.")
        return

    print(f"Running {len(tasks)} MDDM-DMOEA dynamic tasks...")
    
    results = []
    max_workers = min(6, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(future_map):
            results.append(future.result())

    df_new = pd.DataFrame(results)
    
    # Append
    raw_df = pd.concat([df_existing, df_new.drop(columns=["curve"], errors='ignore')], ignore_index=True)
    raw_df["protocol"] = pd.Categorical(raw_df["protocol"], categories=list(PROTOCOLS.keys()), ordered=True)
    raw_df["problem"] = pd.Categorical(raw_df["problem"], categories=list(PROBLEMS), ordered=True)
    
    print("Saving updated asoc_dynamic_raw_metrics.csv")
    raw_df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    run_mddm_append()
