import json
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ============================================================
# Same topology names used in the other multitopology scripts.
# ============================================================
TOPOLOGY_PATHS = {
    "us_ieee118": "data/processed/us_ieee118",
    "uk_ieee30": "data/processed/uk_ieee30",
    "fr_pegase89": "data/processed/fr_pegase89",
    "gb_full": "data/processed/gb_full",
    "eu_case300": "data/processed/eu_case300",
    "india_synthetic": "data/processed/india_synthetic",
}

# ── Vocabulary pools for variety ──────────────────────────
wind_terms = ["high winds", "elevated wind conditions", "strong gusts", "severe wind activity", "storm-force winds"]
outage_verbs = ["tripped", "went offline", "was taken out of service", "experienced a fault", "failed"]
routine_phrases = [
    "System operating within normal parameters.",
    "No anomalies reported during this interval.",
    "Grid conditions remain stable and unremarkable.",
    "Routine operational status confirmed.",
    "All monitored parameters within expected range.",
]

def describe_weather(scenario):
    w = scenario["weather"]
    if w["storm_flag"]:
        term = random.choice(wind_terms)
        return f"{term.capitalize()} recorded at {w['wind_speed_kmh']:.0f} km/h"
    else:
        return f"Wind speed measured at {w['wind_speed_kmh']:.0f} km/h, within typical range"

def describe_event(scenario):
    event = scenario["event"]
    if event["type"] == "none":
        return None
    verb = random.choice(outage_verbs)
    cause = event.get("cause", "unknown cause")
    cause_text = "storm-related conditions" if cause == "storm" else "an equipment failure"
    return f"Line {event['line_id']} {verb} due to {cause_text}"

def generate_operator_alert(scenario):
    """Terse, technical style."""
    parts = [describe_weather(scenario)]
    event_text = describe_event(scenario)
    if event_text:
        parts.append(event_text)
    parts.append(f"Load level at {scenario['load_multiplier']:.2f}x baseline")
    return "Grid alert: " + ". ".join(parts) + "."

def generate_weather_bulletin(scenario):
    """Forecast-style, more narrative."""
    w = scenario["weather"]
    text = f"Weather update: conditions show wind speeds of {w['wind_speed_kmh']:.0f} km/h "
    text += f"and solar irradiance near {w['solar_irradiance_wm2']:.0f} W/m². "
    if w["storm_flag"]:
        text += "Storm activity may affect nearby transmission infrastructure. "
    event_text = describe_event(scenario)
    if event_text:
        text += event_text.capitalize() + " as a result. "
    return text.strip()

def generate_incident_report(scenario):
    """Formal, past-tense, post-incident style."""
    event = scenario["event"]
    if event["type"] == "none":
        return f"At the recorded interval, no incidents were logged. {random.choice(routine_phrases)}"
    cause = event.get("cause", "undetermined cause")
    cause_text = "adverse weather" if cause == "storm" else "a mechanical/equipment issue"
    text = f"Incident summary: line {event['line_id']} was reported out of service, attributed to {cause_text}. "
    text += f"Demand at the time of the event was recorded at {scenario['load_multiplier']:.2f}x typical levels."
    return text

def generate_routine_log(scenario):
    """Dry, minimal, for uneventful scenarios."""
    return random.choice(routine_phrases)

def generate_text_description(scenario):
    """Randomly select a style, weighted toward routine logs for non-events."""
    has_event = scenario["event"]["type"] != "none"

    if not has_event and random.random() < 0.4:
        return generate_routine_log(scenario)

    style = random.choice([
        generate_operator_alert,
        generate_weather_bulletin,
        generate_incident_report,
    ])
    return style(scenario)

def process_split(name, input_path, output_path):
    data = []
    with open(input_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    for scenario in data:
        scenario["text_report"] = generate_text_description(scenario)

    with open(output_path, "w") as f:
        for s in data:
            f.write(json.dumps(s) + "\n")

    print(f"{name}: {len(data)} scenarios with text descriptions saved to {output_path}")


def generate_text_for_topology(topology_name):
    base_dir = TOPOLOGY_PATHS[topology_name]
    input_paths = {
        "train": f"{base_dir}/train.jsonl",
        "val": f"{base_dir}/val.jsonl",
        "test": f"{base_dir}/test.jsonl",
    }
    output_paths = {
        "train": f"{base_dir}/train_multimodal.jsonl",
        "val": f"{base_dir}/val_multimodal.jsonl",
        "test": f"{base_dir}/test_multimodal.jsonl",
    }

    for split_name, input_path in input_paths.items():
        process_split(f"[{topology_name}] {split_name}", input_path, output_paths[split_name])

    print(f"\n[{topology_name}] Done. Sample descriptions:")
    with open(output_paths["train"], "r") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            s = json.loads(line)
            print(f"\n[{s['scenario_id']}] congested={s['congested']}")
            print(f"  \"{s['text_report']}\"")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        topology = sys.argv[1]
    else:
        topology = "uk_ieee30"
    generate_text_for_topology(topology)