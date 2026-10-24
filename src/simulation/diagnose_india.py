import pandapower as pp
import sys
sys.path.insert(0, 'src/simulation')
from build_india_network import build_india_synthetic_network

print("=== Finding max viable load multiplier for india_synthetic ===")
for multiplier in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
    net = build_india_synthetic_network()
    net.load['p_mw'] = net.load['p_mw'] * multiplier
    net.load['q_mvar'] = net.load['q_mvar'] * multiplier
    try:
        pp.runpp(net, enforce_q_lims=False, max_iteration=30, numba=False, init="flat")
        max_loading = net.res_line['loading_percent'].max()
        print(f"multiplier={multiplier}: CONVERGED, max_loading={max_loading:.4f}%")
    except Exception:
        print(f"multiplier={multiplier}: FAILED TO CONVERGE")