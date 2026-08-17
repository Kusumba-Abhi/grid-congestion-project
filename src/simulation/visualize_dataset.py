import json
import matplotlib.pyplot as plt

INPUT_PATH = "data/processed/structured_scenarios.jsonl"

scenarios = []
with open(INPUT_PATH, "r") as f:
    for line in f:
        scenarios.append(json.loads(line))

converged = [s for s in scenarios if s["converged"]]

max_loadings = [s["max_line_loading_pct"] for s in converged]
congested_count = sum(1 for s in converged if s["congested"])
non_congested_count = len(converged) - congested_count
wind_speeds = [s["weather"]["wind_speed_kmh"] for s in scenarios]
load_multipliers = [s["load_multiplier"] for s in scenarios]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].hist(max_loadings, bins=40, color='steelblue', edgecolor='black')
axes[0,0].axvline(100, color='red', linestyle='--', label='Congestion threshold (100%)')
axes[0,0].set_title('Distribution of Max Line Loading (%)')
axes[0,0].set_xlabel('Max Line Loading (%)')
axes[0,0].set_ylabel('Number of Scenarios')
axes[0,0].legend()

axes[0,1].bar(['Non-Congested', 'Congested'], [non_congested_count, congested_count],
              color=['seagreen', 'crimson'])
axes[0,1].set_title(f'Congestion Class Balance ({congested_count}/{len(converged)} = {100*congested_count/len(converged):.1f}%)')
axes[0,1].set_ylabel('Number of Scenarios')

axes[1,0].hist(wind_speeds, bins=30, color='orange', edgecolor='black')
axes[1,0].axvline(65, color='red', linestyle='--', label='Storm threshold (65 km/h)')
axes[1,0].set_title('Wind Speed Distribution Across Scenarios')
axes[1,0].set_xlabel('Wind Speed (km/h)')
axes[1,0].set_ylabel('Number of Scenarios')
axes[1,0].legend()

axes[1,1].hist(load_multipliers, bins=30, color='purple', edgecolor='black')
axes[1,1].set_title('Load Multiplier Distribution')
axes[1,1].set_xlabel('Load Multiplier')
axes[1,1].set_ylabel('Number of Scenarios')

plt.tight_layout()
plt.savefig('experiments/dataset_overview.png', dpi=150)
print("Saved plot to experiments/dataset_overview.png")
plt.show()