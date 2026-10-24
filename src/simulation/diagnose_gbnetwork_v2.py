import pandapower as pp
import pandapower.networks as pn

# From diagnostic v1: multiplier=1.3 is the highest that converges, giving 60.9% max loading
max_loading_at_derate1 = 60.9552
safe_multiplier = 1.3
target_loading = 140.0

calculated_derate = max_loading_at_derate1 / target_loading
print(f"Calculated derate for ~{target_loading}% target at {safe_multiplier}x load: {calculated_derate:.6f}")

print("\n=== Verifying with calculated derate ===")
net = pn.GBnetwork()
net.line['max_i_ka'] = net.line['max_i_ka'] * calculated_derate
net.load['p_mw'] = net.load['p_mw'] * safe_multiplier
net.load['q_mvar'] = net.load['q_mvar'] * safe_multiplier
try:
    pp.runpp(net, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
    print(f"Max loading at {safe_multiplier}x load, calculated derate: {net.res_line['loading_percent'].max():.4f}%")
except Exception:
    print(f"FAILED TO CONVERGE even with derate applied -- derate may be too aggressive")