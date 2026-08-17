import json
import matplotlib.pyplot as plt
import re
from collections import Counter

# ── Model comparison chart ──────────────────────────────
model1_metrics = {"Accuracy": 0.984, "Precision": 0.915, "Recall": 1.000, "F1 Score": 0.956}
model2_metrics = {"Accuracy": 0.991, "Precision": 0.981, "Recall": 0.963, "F1 Score": 0.972}

labels = list(model1_metrics.keys())
m1_values = list(model1_metrics.values())
m2_values = list(model2_metrics.values())

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
bars1 = ax.bar([i - width/2 for i in x], m1_values, width, label='Model 1 (Structured Only)', color='steelblue')
bars2 = ax.bar([i + width/2 for i in x], m2_values, width, label='Model 2 (Structured + Text)', color='crimson')

ax.set_ylabel('Score')
ax.set_title('Model 1 vs Model 2 Performance Comparison (Test Set)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.8, 1.05)
ax.legend()
ax.bar_label(bars1, fmt='%.3f', padding=3)
ax.bar_label(bars2, fmt='%.3f', padding=3)

plt.tight_layout()
plt.savefig('experiments/model_comparison.png', dpi=150)
print("Saved model comparison chart to experiments/model_comparison.png")

# ── Text/unstructured dataset statistics ─────────────────
with open("data/processed/train_multimodal.jsonl", "r") as f:
    train_multimodal = [json.loads(line) for line in f]

text_lengths = [len(re.findall(r"[a-zA-Z]+", s["text_report"])) for s in train_multimodal]

fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(text_lengths, bins=20, color='mediumpurple', edgecolor='black')
axes[0].set_title('Text Report Length Distribution (words)')
axes[0].set_xlabel('Number of Words')
axes[0].set_ylabel('Number of Scenarios')

# Rough style classification based on opening phrase
style_counts = Counter()
for s in train_multimodal:
    text = s["text_report"]
    if text.startswith("Grid alert"):
        style_counts["Operator Alert"] += 1
    elif text.startswith("Weather update"):
        style_counts["Weather Bulletin"] += 1
    elif text.startswith("Incident summary") or text.startswith("At the recorded"):
        style_counts["Incident Report"] += 1
    else:
        style_counts["Routine Log"] += 1

axes[1].bar(style_counts.keys(), style_counts.values(), color='teal')
axes[1].set_title('Text Style Distribution')
axes[1].set_ylabel('Number of Scenarios')
plt.setp(axes[1].get_xticklabels(), rotation=20, ha='right')

plt.tight_layout()
plt.savefig('experiments/text_dataset_overview.png', dpi=150)
print("Saved text dataset overview to experiments/text_dataset_overview.png")

print(f"\nVocabulary size: 98 words")
print(f"Average text length: {sum(text_lengths)/len(text_lengths):.1f} words")
print(f"Text style breakdown: {dict(style_counts)}")