import pandapower as pp
import pandapower.networks as pn

# From diagnostic v1: only multiplier=1.0 converges, giving 8.7325% max loading
max_loading_at_derate1 = 8.7325
safe_multiplier = 1.0
target_loading = 150.0

calculated_derate = max_loading_at_derate1 / target_loading
print(f"Calculated derate for ~{target_loading}% target at {safe_multiplier}x load: {calculated_derate:.6f}")

print("\n=== Verifying with calculated derate ===")
net = pn.case300()
net.line['max_i_ka'] = net.line['max_i_ka'] * calculated_derate
net.load['p_mw'] = net.load['p_mw'] * safe_multiplier
net.load['q_mvar'] = net.load['q_mvar'] * safe_multiplier
try:
    pp.runpp(net, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
    print(f"Max loading at {safe_multiplier}x load, calculated derate: {net.res_line['loading_percent'].max():.4f}%")
except Exception:
    print(f"FAILED TO CONVERGE even with derate applied -- derate may be too aggressive")

print("\n=== Also checking a slightly lower multiplier (0.85x) with same derate ===")
net2 = pn.case300()
net2.line['max_i_ka'] = net2.line['max_i_ka'] * calculated_derate
net2.load['p_mw'] = net2.load['p_mw'] * 0.85
net2.load['q_mvar'] = net2.load['q_mvar'] * 0.85
try:
    pp.runpp(net2, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
    print(f"Max loading at 0.85x load, calculated derate: {net2.res_line['loading_percent'].max():.4f}%")
except Exception:
    print(f"FAILED TO CONVERGE at 0.85x")