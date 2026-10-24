import json
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# ============================================================
# Same topology names used in generate_scenarios_multitopology.py
# Each maps to the folder created by that script.
# ============================================================
TOPOLOGY_PATHS = {
    "us_ieee118": "data/processed/us_ieee118",
    "uk_ieee30": "data/processed/uk_ieee30",
    "fr_pegase89": "data/processed/fr_pegase89",
    "gb_full": "data/processed/gb_full",
    "eu_case300": "data/processed/eu_case300",
    "india_synthetic": "data/processed/india_synthetic",
}


def split_dataset_for_topology(topology_name):
    base_dir = TOPOLOGY_PATHS[topology_name]
    input_path = f"{base_dir}/structured_scenarios.jsonl"
    train_path = f"{base_dir}/train.jsonl"
    val_path = f"{base_dir}/val.jsonl"
    test_path = f"{base_dir}/test.jsonl"

    # Load all scenarios
    scenarios = []
    with open(input_path, "r") as f:
        for line in f:
            scenarios.append(json.loads(line))

    print(f"[{topology_name}] Total scenarios loaded: {len(scenarios)}")

    # Keep only converged scenarios (non-converged have no valid label)
    converged = [s for s in scenarios if s["converged"]]
    print(f"[{topology_name}] Converged scenarios: {len(converged)}")

    # Labels for stratification
    labels = [s["congested"] for s in converged]

    # First split: 70% train, 30% temp (which becomes val+test)
    train, temp = train_test_split(
        converged, test_size=0.30, stratify=labels, random_state=RANDOM_SEED
    )

    # Second split: split temp into 15% val, 15% test (50/50 of the 30%)
    temp_labels = [s["congested"] for s in temp]
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp_labels, random_state=RANDOM_SEED
    )

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    save_jsonl(train, train_path)
    save_jsonl(val, val_path)
    save_jsonl(test, test_path)

    def congestion_rate(data):
        return sum(1 for d in data if d["congested"]) / len(data) * 100

    print(f"[{topology_name}] Train: {len(train)} scenarios, {congestion_rate(train):.1f}% congested")
    print(f"[{topology_name}] Val:   {len(val)} scenarios, {congestion_rate(val):.1f}% congested")
    print(f"[{topology_name}] Test:  {len(test)} scenarios, {congestion_rate(test):.1f}% congested")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        topology = sys.argv[1]
    else:
        topology = "uk_ieee30"  # default if no argument given
    split_dataset_for_topology(topology)