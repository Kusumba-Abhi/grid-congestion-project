import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import json
import random
from datetime import datetime, timedelta
from build_india_network import build_india_synthetic_network

# ============================================================
# TOPOLOGY CONFIGS
# Add one entry per grid you want to support. Every number here
# is something you may need to tune per-topology to get a
# realistic (not 0% or 100%) congestion rate -- same way the
# original 0.07 derating factor was tuned for case118.
# ============================================================

TOPOLOGY_CONFIGS = {
    "us_ieee118": {
        "loader": pn.case118,
        "thermal_derate": 0.07,
        "n_renewable_buses": 10,
        "wind_mw_range": (40, 120),
        "solar_mw_range": (20, 80),
        "load_multiplier_range": (0.8, 2.2),
        "output_path": "data/processed/us_ieee118/structured_scenarios.jsonl",
    },
    "uk_ieee30": {
        "loader": pn.case_ieee30,      # small, fast to calibrate -- do this one first
        "thermal_derate": 0.00001,     # 0.00002->4%, 0.000006->62.3%, 0.000002->100%; narrowing toward ~30-40%
        "n_renewable_buses": 4,        # ~30% of buses, scaled down from 118-bus's 10/118
        "wind_mw_range": (40, 120),    # placeholder -- needs recalibration
        "solar_mw_range": (20, 80),    # placeholder -- needs recalibration
        "load_multiplier_range": (0.8, 2.2),
        "output_path": "data/processed/uk_ieee30/structured_scenarios.jsonl",
    },
    "fr_pegase89": {
        "loader": pn.case89pegase,     # French PEGASE network, 89 buses
        "thermal_derate": 1.0,         # diagnostic: baseline already 96.4% loaded, near-realistic by default -- no derate needed
        "n_renewable_buses": 8,
        "wind_mw_range": (40, 120),    # placeholder -- needs recalibration
        "solar_mw_range": (20, 80),    # placeholder -- needs recalibration
        "load_multiplier_range": (0.7, 1.15),  # diagnostic: 1.1x already >100% loaded, 2.2x fails to converge entirely
        "output_path": "data/processed/fr_pegase89/structured_scenarios.jsonl",
    },
    "gb_full": {
        "loader": pn.GBnetwork,        # Great Britain full transmission network
        "thermal_derate": 0.435394,    # diagnostic: 1.3x load (max viable) gave 60.96% loading uncongested; calculated for ~140% target
        "n_renewable_buses": 15,
        "wind_mw_range": (40, 120),    # placeholder -- needs recalibration
        "solar_mw_range": (20, 80),    # placeholder -- needs recalibration
        "load_multiplier_range": (0.7, 1.3),   # diagnostic: 1.5x+ fails to converge for this network
        "output_path": "data/processed/gb_full/structured_scenarios.jsonl",
    },
    "eu_case300": {
        "loader": pn.case300,
        "thermal_derate": 0.111955,        # placeholder -- needs recalibration
        "n_renewable_buses": 15,
        "wind_mw_range": (40, 120),    # placeholder -- needs recalibration
        "solar_mw_range": (20, 80),    # placeholder -- needs recalibration
        "load_multiplier_range": (0.95, 1.02),
        "output_path": "data/processed/eu_case300/structured_scenarios.jsonl",
    },
        "india_synthetic": {
        "loader": build_india_synthetic_network,  # custom-built 5-region network, parameterized from real CEA/Grid-India data
        "thermal_derate": 1.0,         # diagnostic: real corridor capacities already produce realistic loading, no extra derate needed
        "n_renewable_buses": 4,        # matches the 4 non-slack regional gen buses in this network
        "wind_mw_range": (40, 120),    # placeholder -- needs recalibration
        "solar_mw_range": (20, 80),    # placeholder -- needs recalibration
        "load_multiplier_range": (0.9, 1.1),  # diagnostic: smooth, realistic loading range; below 0.8 or above 1.2 behaves oddly/fails
        "output_path": "data/processed/india_synthetic/structured_scenarios.jsonl",
    },
}

N_SCENARIOS = 2500
RANDOM_SEED = 42


def generate_dataset_for_topology(topology_name):
    cfg = TOPOLOGY_CONFIGS[topology_name]
    print(f"[{topology_name}] Using thermal_derate = {cfg['thermal_derate']}")  # DEBUG -- confirms which value is active

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    base_net = cfg["loader"]()
    gen_buses = base_net.gen['bus'].tolist()
    renewable_buses = random.sample(gen_buses, min(cfg["n_renewable_buses"], len(gen_buses)))
    line_ids = base_net.line.index.tolist()

    def generate_scenario(scenario_id, timestamp):
        net = cfg["loader"]()
        net.line['max_i_ka'] = net.line['max_i_ka'] * cfg["thermal_derate"]

        wind_speed = np.random.uniform(0, 90)
        solar_irradiance = np.random.uniform(0, 1000)
        storm_flag = wind_speed > 65

        renewable_output = {}
        for bus in renewable_buses:
            wind_mw = max(0, (wind_speed / 90) * np.random.uniform(*cfg["wind_mw_range"]))
            solar_mw = max(0, (solar_irradiance / 1000) * np.random.uniform(*cfg["solar_mw_range"]))
            renewable_output[bus] = {"wind_mw": round(wind_mw, 2), "solar_mw": round(solar_mw, 2)}
            pp.create_sgen(net, bus=bus, p_mw=wind_mw + solar_mw, name=f"renewable_bus{bus}")

        load_multiplier = np.random.uniform(*cfg["load_multiplier_range"])
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
            "topology": topology_name,
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
            print(f"[{topology_name}] Generated {i + 1}/{N_SCENARIOS} scenarios...")

    with open(cfg["output_path"], "w") as f:
        for s in scenarios:
            f.write(json.dumps(s) + "\n")

    converged_count = sum(1 for s in scenarios if s["converged"])
    congested_count = sum(1 for s in scenarios if s["congested"])

    print(f"\n[{topology_name}] Done. Saved {len(scenarios)} scenarios to {cfg['output_path']}")
    print(f"[{topology_name}] Converged: {converged_count}/{N_SCENARIOS}")
    print(f"[{topology_name}] Congested: {congested_count}/{converged_count if converged_count else 1} "
          f"({100 * congested_count / max(converged_count,1):.1f}%)")

    return scenarios


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        topology = sys.argv[1]
    else:
        topology = "fr_pegase89"  # default if no argument given
    generate_dataset_for_topology(topology)