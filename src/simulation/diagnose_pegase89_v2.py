import pandapower as pp
import pandapower.networks as pn

print("=== case89pegase: finding max viable load multiplier ===")

for multiplier in [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.2]:
    net = pn.case89pegase()
    net.load['p_mw'] = net.load['p_mw'] * multiplier
    net.load['q_mvar'] = net.load['q_mvar'] * multiplier
    try:
        pp.runpp(net, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
        max_loading = net.res_line['loading_percent'].max()
        print(f"multiplier={multiplier}: CONVERGED, max_loading={max_loading:.4f}%")
    except Exception as e:
        print(f"multiplier={multiplier}: FAILED TO CONVERGE")