"""
Synthetic representative network for the Indian National Grid.

STRUCTURE: synthetic (constructed by us) -- India's real transmission
topology is not publicly available, for the same infrastructure-security
reasons true of every country.

PARAMETERS: real, from published sources --
  - Generation capacity by region and type: CEA regional installed
    capacity report, as of 30 June 2026
    (https://cea.nic.in/wp-content/uploads/installed/2026/06/Website_June.pdf)
  - Inter-regional transfer capacity: CEA National Electricity Plan
    Volume II (Transmission) and Grid-India ATC reporting; two corridors
    (WR-NR, WR-SR) have specific published figures, the remaining
    corridors are estimated proportionally from regional generation
    scale since corridor-specific figures were not found for all links --
    this is noted explicitly and should be stated as an assumption in
    the paper, not presented as an exact published figure.

This is a 5-bus aggregate representation: one bus per region. Each bus
carries the region's total conventional (thermal+nuclear+hydro) and
renewable (wind+solar+other) capacity as separate generator elements,
and a load representing regional demand.
"""

import pandapower as pp
import numpy as np

# ============================================================
# REAL DATA: regional generation capacity (MW), CEA, 30-06-2026
# ============================================================
REGIONS = {
    "NR": {"name": "Northern",     "conventional_mw": 69553 + 2220 + 22960, "renewable_mw": 66555},
    "WR": {"name": "Western",      "conventional_mw": 84212 + 3240 + 8094,  "renewable_mw": 92531},
    "SR": {"name": "Southern",     "conventional_mw": 54789 + 3320 + 13596, "renewable_mw": 73041},
    "ER": {"name": "Eastern",      "conventional_mw": 41938 + 0 + 5988,     "renewable_mw": 3237},
    "NER": {"name": "North-Eastern", "conventional_mw": 2451 + 0 + 3023,    "renewable_mw": 1160},
}

# Assumed regional peak demand as a fraction of installed capacity.
# India's all-India peak demand vs installed capacity ratio is
# roughly 50-55% (peak demand ~250 GW vs ~550 GW installed, 2026);
# applied uniformly per region as a simplifying assumption.
DEMAND_FACTOR = 0.55

# ============================================================
# REAL DATA (partial) + ESTIMATED: inter-regional transfer capacity (MW)
# Published: WR-NR and WR-SR. Remaining corridors estimated
# proportionally to connected regions' generation scale -- ASSUMPTION,
# not a published figure. Documented here for transparency.
# ============================================================
CORRIDORS = [
    # (region_a, region_b, capacity_mw, source)
    ("WR", "NR", 56720, "published (NEP Vol II, target FY27)"),
    ("WR", "SR", 28120, "published (NEP Vol II, target FY27)"),
    ("WR", "ER", 12000, "estimated -- proportional to WR/ER generation scale"),
    ("NR", "ER", 10000, "estimated -- proportional to NR/ER generation scale"),
    ("ER", "NER", 3000, "estimated -- NER is the smallest region by far"),
    ("SR", "ER", 5000, "estimated -- secondary SR-ER link for mesh connectivity"),
]

NOMINAL_KV = 400  # standard Indian inter-regional transmission voltage tier


def build_india_synthetic_network():
    net = pp.create_empty_network(name="india_synthetic_5region")

    bus_idx = {}
    for region_code, info in REGIONS.items():
        bus_idx[region_code] = pp.create_bus(
            net, vn_kv=NOMINAL_KV, name=f"{info['name']} Region"
        )

    # Slack bus: Western region (largest, most central to the corridors)
    pp.create_ext_grid(net, bus=bus_idx["WR"], vm_pu=1.0, name="WR Slack")

    # Conventional generation per region (scaled to MW, /1000 kept as MW directly)
    for region_code, info in REGIONS.items():
        if region_code == "WR":
            continue  # WR's conventional capacity backs the slack bus implicitly
        pp.create_gen(
            net, bus=bus_idx[region_code], p_mw=info["conventional_mw"] * 0.6,
            vm_pu=1.0, name=f"{region_code}_conventional", controllable=True,
            max_p_mw=info["conventional_mw"], min_p_mw=0
        )

    # Renewable generation per region (as static generators -- variable output)
    for region_code, info in REGIONS.items():
        pp.create_sgen(
            net, bus=bus_idx[region_code], p_mw=info["renewable_mw"] * 0.3,
            name=f"{region_code}_renewable"
        )

    # Load per region
    for region_code, info in REGIONS.items():
        total_capacity = info["conventional_mw"] + info["renewable_mw"]
        pp.create_load(
            net, bus=bus_idx[region_code], p_mw=total_capacity * DEMAND_FACTOR,
            q_mvar=total_capacity * DEMAND_FACTOR * 0.3,  # assumed power factor ~0.96
            name=f"{region_code}_load"
        )

    # Inter-regional lines, rated from real/estimated transfer capacities.
    # NOTE: each real corridor is actually many parallel physical circuits
    # (this is how India achieves e.g. 56,720 MW on the WR-NR corridor).
    # Modeling it as one line with realistic per-km impedance over a
    # realistic distance (~300km) gives an unrealistically high total
    # impedance for that much current, and power flow fails to converge.
    # We use a short effective length to represent the LOW aggregate
    # impedance of many parallel circuits combined, while max_i_ka still
    # represents the real aggregate current-carrying (MW) capacity.
    for region_a, region_b, capacity_mw, source in CORRIDORS:
        max_i_ka = capacity_mw / (np.sqrt(3) * NOMINAL_KV)
        pp.create_line_from_parameters(
            net, from_bus=bus_idx[region_a], to_bus=bus_idx[region_b],
            length_km=20,  # short effective length -- represents combined low impedance of many parallel real circuits
            r_ohm_per_km=0.03, x_ohm_per_km=0.32, c_nf_per_km=9.0,
            max_i_ka=max_i_ka,
            name=f"{region_a}-{region_b} corridor ({source})"
        )

    return net


if __name__ == "__main__":
    net = build_india_synthetic_network()
    print("=== india_synthetic network built ===")
    print(f"Buses: {len(net.bus)}")
    print(f"Lines: {len(net.line)}")
    print(f"Generators: {len(net.gen)}")
    print(f"Static generators (renewable): {len(net.sgen)}")
    print(f"Loads: {len(net.load)}")
    print(f"External grid (slack): {len(net.ext_grid)}")

    print("\n=== Running power flow ===")
    pp.runpp(net, enforce_q_lims=False, max_iteration=30, numba=False, init="flat")
    print(f"Converged. Max line loading: {net.res_line['loading_percent'].max():.4f}%")
    print("\nPer-line loading:")
    for idx, row in net.res_line.iterrows():
        print(f"  {net.line.at[idx, 'name']}: {row['loading_percent']:.2f}%")