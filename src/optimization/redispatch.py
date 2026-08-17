import pandapower as pp
import pandapower.networks as pn
import json
import numpy as np

TEST_PATH = "data/processed/test_multimodal.jsonl"
OUTPUT_PATH = "experiments/redispatch_results.txt"

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def rebuild_network(scenario):
    net = pn.case118()
    net.line['max_i_ka'] = net.line['max_i_ka'] * 0.07

    for bus_str, values in scenario["renewable_output"].items():
        bus = int(bus_str)
        total_mw = values["wind_mw"] + values["solar_mw"]
        max_cap = max(total_mw * 1.5, 10)
        pp.create_sgen(net, bus=bus, p_mw=total_mw, name=f"renewable_bus{bus}", controllable=True,
                        min_p_mw=0, max_p_mw=max_cap,
                        min_q_mvar=-max_cap * 0.3, max_q_mvar=max_cap * 0.3)

    net.load['p_mw'] = net.load['p_mw'] * scenario["load_multiplier"]
    net.load['q_mvar'] = net.load['q_mvar'] * scenario["load_multiplier"]

    if scenario["event"]["type"] == "line_outage":
        net.line.at[scenario["event"]["line_id"], 'in_service'] = False

    return net

def setup_generator_costs(net):
    net.poly_cost = net.poly_cost.iloc[0:0]

    for idx in net.gen.index:
        net.gen.at[idx, 'controllable'] = True
        net.gen.at[idx, 'min_p_mw'] = 0
        net.gen.at[idx, 'max_p_mw'] = max(net.gen.at[idx, 'max_p_mw'], net.gen.at[idx, 'p_mw'] * 1.2 + 1)
        if 'min_q_mvar' not in net.gen.columns or pd_isna(net.gen.at[idx, 'min_q_mvar']):
            net.gen.at[idx, 'min_q_mvar'] = -net.gen.at[idx, 'max_p_mw'] * 0.5
        if 'max_q_mvar' not in net.gen.columns or pd_isna(net.gen.at[idx, 'max_q_mvar']):
            net.gen.at[idx, 'max_q_mvar'] = net.gen.at[idx, 'max_p_mw'] * 0.5
        pp.create_poly_cost(net, idx, 'gen', cp1_eur_per_mw=100)

    for idx in net.sgen.index:
        pp.create_poly_cost(net, idx, 'sgen', cp1_eur_per_mw=5)

def pd_isna(val):
    import pandas as pd
    return pd.isna(val)

def attempt_redispatch(scenario):
    net = rebuild_network(scenario)
    setup_generator_costs(net)

    try:
        pp.runopp(net, numba=False)
        max_loading_after = net.res_line['loading_percent'].max()
        resolved = bool(max_loading_after <= 100)
        return {
            "scenario_id": scenario["scenario_id"],
            "converged": True,
            "max_loading_before": scenario["max_line_loading_pct"],
            "max_loading_after": round(max_loading_after, 2),
            "resolved": resolved,
        }
    except Exception as e:
        return {
            "scenario_id": scenario["scenario_id"],
            "converged": False,
            "max_loading_before": scenario["max_line_loading_pct"],
            "max_loading_after": None,
            "resolved": False,
            "error": str(e)[:150]
        }

test_data = load_jsonl(TEST_PATH)
congested_scenarios = [s for s in test_data if s["congested"]]

print(f"Found {len(congested_scenarios)} congested scenarios in test set")
print("Running OPF-based redispatch on each...\n")

results = []
for i, scenario in enumerate(congested_scenarios):
    result = attempt_redispatch(scenario)
    results.append(result)
    status = "RESOLVED" if result["resolved"] else ("FAILED" if not result["converged"] else "NOT RESOLVED")
    print(f"[{i+1}/{len(congested_scenarios)}] {scenario['scenario_id']}: {status} "
          f"(before: {result['max_loading_before']}%, after: {result['max_loading_after']}%)")
    if not result["converged"]:
        print(f"  Error: {result.get('error', 'unknown')}")

converged_count = sum(1 for r in results if r["converged"])
resolved_count = sum(1 for r in results if r["resolved"])

summary = f"""
REDISPATCH OPTIMIZATION RESULTS
Congested test scenarios: {len(congested_scenarios)}
OPF converged: {converged_count}/{len(congested_scenarios)}
Congestion resolved: {resolved_count}/{converged_count if converged_count else 1}

Individual results:
"""
for r in results:
    summary += f"  {r['scenario_id']}: before={r['max_loading_before']}%, after={r['max_loading_after']}%, resolved={r['resolved']}\n"

with open(OUTPUT_PATH, "w") as f:
    f.write(summary)

print(f"\n\nSummary: {resolved_count}/{converged_count if converged_count else 1} congested scenarios successfully resolved via redispatch")
print(f"Results saved to {OUTPUT_PATH}")