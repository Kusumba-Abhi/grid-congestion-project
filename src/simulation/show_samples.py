import json

INPUT_PATH = "data/processed/structured_scenarios.jsonl"

scenarios = []
with open(INPUT_PATH, "r") as f:
    for line in f:
        scenarios.append(json.loads(line))

# Find one congested and one non-congested example to showcase
congested_example = next(s for s in scenarios if s["congested"] == True)
normal_example = next(s for s in scenarios if s["congested"] == False)

def pretty_print(s, label):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Scenario ID:        {s['scenario_id']}")
    print(f"Timestamp:          {s['timestamp']}")
    print(f"Wind Speed:         {s['weather']['wind_speed_kmh']} km/h")
    print(f"Solar Irradiance:   {s['weather']['solar_irradiance_wm2']} W/m^2")
    print(f"Storm Flag:         {s['weather']['storm_flag']}")
    print(f"Load Multiplier:    {s['load_multiplier']}x")
    print(f"Event:              {s['event']['type']} (cause: {s['event'].get('cause', 'N/A')})")
    print(f"Converged:          {s['converged']}")
    print(f"Max Line Loading:   {s['max_line_loading_pct']}%")
    print(f"Congested:          {s['congested']}")
    print(f"Overloaded Lines:   {s['overloaded_lines']}")

pretty_print(congested_example, "EXAMPLE: CONGESTED SCENARIO")
pretty_print(normal_example, "EXAMPLE: NON-CONGESTED SCENARIO")