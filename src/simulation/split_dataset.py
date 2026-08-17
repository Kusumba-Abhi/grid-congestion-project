import json
import numpy as np
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/processed/structured_scenarios.jsonl"
TRAIN_PATH = "data/processed/train.jsonl"
VAL_PATH = "data/processed/val.jsonl"
TEST_PATH = "data/processed/test.jsonl"

RANDOM_SEED = 42

# Load all scenarios
scenarios = []
with open(INPUT_PATH, "r") as f:
    for line in f:
        scenarios.append(json.loads(line))

print(f"Total scenarios loaded: {len(scenarios)}")

# Keep only converged scenarios (non-converged have no valid label)
converged = [s for s in scenarios if s["converged"]]
print(f"Converged scenarios: {len(converged)}")

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

save_jsonl(train, TRAIN_PATH)
save_jsonl(val, VAL_PATH)
save_jsonl(test, TEST_PATH)

def congestion_rate(data):
    return sum(1 for d in data if d["congested"]) / len(data) * 100

print(f"\nTrain: {len(train)} scenarios, {congestion_rate(train):.1f}% congested")
print(f"Val:   {len(val)} scenarios, {congestion_rate(val):.1f}% congested")
print(f"Test:  {len(test)} scenarios, {congestion_rate(test):.1f}% congested")