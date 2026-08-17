import json

STRUCTURED_PATH = "data/processed/structured_scenarios.jsonl"
MULTIMODAL_PATH = "data/processed/train_multimodal.jsonl"

# ── Load structured-only data ──────────────────────────
with open(STRUCTURED_PATH, "r") as f:
    structured_scenarios = [json.loads(line) for line in f]

# Pick one congested and one non-congested example
structured_congested = next(s for s in structured_scenarios if s["congested"] == True)
structured_normal = next(s for s in structured_scenarios if s["congested"] == False)

structured_output = {
    "congested_example": structured_congested,
    "non_congested_example": structured_normal
}

with open("experiments/sample_structured_scenarios.json", "w") as f:
    json.dump(structured_output, f, indent=4)

print("Saved structured examples to experiments/sample_structured_scenarios.json")

# ── Load multimodal (structured + text) data ────────────
with open(MULTIMODAL_PATH, "r") as f:
    multimodal_scenarios = [json.loads(line) for line in f]

multimodal_congested = next(s for s in multimodal_scenarios if s["congested"] == True)
multimodal_normal = next(s for s in multimodal_scenarios if s["congested"] == False)

multimodal_output = {
    "congested_example": multimodal_congested,
    "non_congested_example": multimodal_normal
}

with open("experiments/sample_multimodal_scenarios.json", "w") as f:
    json.dump(multimodal_output, f, indent=4)

print("Saved multimodal examples to experiments/sample_multimodal_scenarios.json")

# ── Print both to terminal for immediate viewing ─────────
print("\n" + "="*60)
print("STRUCTURED-ONLY EXAMPLE (Congested)")
print("="*60)
print(json.dumps(structured_congested, indent=4))

print("\n" + "="*60)
print("MULTIMODAL EXAMPLE (Congested) — includes text_report")
print("="*60)
print(json.dumps(multimodal_congested, indent=4))