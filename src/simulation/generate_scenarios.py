import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import json
import random
from datetime import datetime, timedelta

N_SCENARIOS = 2500
RANDOM_SEED = 42
OUTPUT_PATH = "data/processed/structured_scenarios.jsonl"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

base_net = pn.case118()
gen_buses = base_net.gen['bus'].tolist()
renewable_buses = random.sample(gen_buses, min(10, len(gen_buses)))
line_ids = base_net.line.index.tolist()

def generate_scenario(scenario_id, timestamp):
    net = pn.case118()
    net.line['max_i_ka'] = net.line['max_i_ka'] * 0.07

    wind_speed = np.random.uniform(0, 90)
    solar_irradiance = np.random.uniform(0, 1000)
    storm_flag = wind_speed > 65

    renewable_output = {}
    for bus in renewable_buses:
        wind_mw = max(0, (wind_speed / 90) * np.random.uniform(40, 120))
        solar_mw = max(0, (solar_irradiance / 1000) * np.random.uniform(20, 80))
        renewable_output[bus] = {"wind_mw": round(wind_mw, 2), "solar_mw": round(solar_mw, 2)}
        pp.create_sgen(net, bus=bus, p_mw=wind_mw + solar_mw, name=f"renewable_bus{bus}")

    load_multiplier = np.random.uniform(0.8, 2.2)
    net.load['p_mw'] = net.load['p_mw'] * load_multiplier
    net.load['q_mvar'] = net.load['q_mvar'] * load_multiplier

    event = {"type": "none"}
    outage_line = None
    if storm_flag and np.random.rand() < 0.7:
        outage_line = random.choice(line_ids)
        cause = "storm"
    elif np.random.rand() < 0.25:
        outage_line = random.choice(line_ids)
        cause = "equipment_failure"

    if outage_line is not None:
        net.line.at[outage_line, 'in_service'] = False
        event = {"type": "line_outage", "line_id": int(outage_line), "cause": cause}

    try:
        pp.runpp(net, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
        converged = True
        max_loading = net.res_line['loading_percent'].max()
        congested = bool(max_loading > 100)
        overloaded_lines = net.res_line[net.res_line['loading_percent'] > 100].index.tolist()
    except Exception:
        converged = False
        max_loading = None
        congested = None
        overloaded_lines = []

    return {
        "scenario_id": f"SC_{scenario_id:05d}",
        "timestamp": timestamp.isoformat(),
        "weather": {
            "wind_speed_kmh": round(wind_speed, 2),
            "solar_irradiance_wm2": round(solar_irradiance, 2),
            "storm_flag": storm_flag
        },
        "renewable_output": renewable_output,
        "load_multiplier": round(load_multiplier, 3),
        "event": event,
        "converged": converged,
        "max_line_loading_pct": round(max_loading, 2) if max_loading is not None else None,
        "congested": congested,
        "overloaded_lines": overloaded_lines
    }

start_time = datetime(2025, 1, 1)
scenarios = []

for i in range(N_SCENARIOS):
    ts = start_time + timedelta(hours=i)
    scenarios.append(generate_scenario(i, ts))
    if (i + 1) % 250 == 0:
        print(f"Generated {i + 1}/{N_SCENARIOS} scenarios...")

with open(OUTPUT_PATH, "w") as f:
    for s in scenarios:
        f.write(json.dumps(s) + "\n")

converged_count = sum(1 for s in scenarios if s["converged"])
congested_count = sum(1 for s in scenarios if s["congested"])

print(f"\nDone. Saved {len(scenarios)} scenarios to {OUTPUT_PATH}")
print(f"Converged: {converged_count}/{N_SCENARIOS}")
print(f"Congested: {congested_count}/{converged_count if converged_count else 1} "
      f"({100 * congested_count / max(converged_count,1):.1f}%)")