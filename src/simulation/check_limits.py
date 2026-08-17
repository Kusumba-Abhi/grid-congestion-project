import pandapower as pp
import pandapower.networks as pn

for limit_factor in [0.05, 0.04, 0.03, 0.02]:
    net = pn.case118()
    net.line['max_i_ka'] = net.line['max_i_ka'] * limit_factor
    net.load['p_mw'] = net.load['p_mw'] * 1.4
    net.load['q_mvar'] = net.load['q_mvar'] * 1.4

    try:
        pp.runpp(net, enforce_q_lims=False, max_iteration=20, numba=False)
        max_loading = net.res_line['loading_percent'].max()
        congested = (net.res_line['loading_percent'] > 100).sum()
        print(f"Thermal limit factor {limit_factor}: max_loading={max_loading:.1f}%, congested_lines={congested}/173, CONVERGED")
    except Exception:
        print(f"Thermal limit factor {limit_factor}: FAILED TO CONVERGE")