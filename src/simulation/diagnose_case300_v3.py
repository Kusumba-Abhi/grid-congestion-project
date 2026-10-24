import pandapower as pp
import pandapower.networks as pn

calculated_derate = 0.058217

print("=== Testing narrow multiplier range close to 1.0 ===")
for multiplier in [0.9, 0.95, 0.98, 1.0, 1.02, 1.05]:
    net = pn.case300()
    net.line['max_i_ka'] = net.line['max_i_ka'] * calculated_derate
    net.load['p_mw'] = net.load['p_mw'] * multiplier
    net.load['q_mvar'] = net.load['q_mvar'] * multiplier
    try:
        pp.runpp(net, enforce_q_lims=False, max_iteration=30, numba=False, init="flat")
        max_loading = net.res_line['loading_percent'].max()
        print(f"multiplier={multiplier}: CONVERGED, max_loading={max_loading:.4f}%")
    except Exception:
        print(f"multiplier={multiplier}: FAILED TO CONVERGE")