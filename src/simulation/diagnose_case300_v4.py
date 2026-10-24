import pandapower as pp
import pandapower.networks as pn

max_loading_at_derate1 = 8.7325
target_baseline_loading = 78.0

new_derate = max_loading_at_derate1 / target_baseline_loading
print(f"New calculated derate for ~{target_baseline_loading}% baseline: {new_derate:.6f}")

print("\n=== Verifying across the known-safe multiplier range ===")
for multiplier in [0.95, 0.98, 1.0, 1.02]:
    net = pn.case300()
    net.line['max_i_ka'] = net.line['max_i_ka'] * new_derate
    net.load['p_mw'] = net.load['p_mw'] * multiplier
    net.load['q_mvar'] = net.load['q_mvar'] * multiplier
    try:
        pp.runpp(net, enforce_q_lims=False, max_iteration=30, numba=False, init="flat")
        max_loading = net.res_line['loading_percent'].max()
        print(f"multiplier={multiplier}: CONVERGED, max_loading={max_loading:.4f}%")
    except Exception:
        print(f"multiplier={multiplier}: FAILED TO CONVERGE")