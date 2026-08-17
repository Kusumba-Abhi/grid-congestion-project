import json

INPUT_PATH = "data/processed/structured_scenarios.jsonl"

# Change this to view a different scenario (0 = first one, 1 = second, etc.)
SCENARIO_INDEX = 0

with open(INPUT_PATH, "r") as f:
    lines = f.readlines()

scenario = json.loads(lines[SCENARIO_INDEX])

# Pretty-print with indentation
print(json.dumps(scenario, indent=4))