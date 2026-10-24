import pandapower as pp
import pandapower.networks as pn

net = pn.case89pegase()

print("=== case89pegase basic info ===")
print(f"Buses: {len(net.bus)}")
print(f"Lines: {len(net.line)}")
print(f"Generators: {len(net.gen)}")
print(f"Loads: {len(net.load)}")

print("\n=== Running power flow with NO derating, NO extra load ===")
pp.runpp(net, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
print(f"Max loading at baseline: {net.res_line['loading_percent'].max():.6f}%")

print("\n=== Now with load multiplier x2.2 (our standard top of range), no derate ===")
net2 = pn.case89pegase()
net2.load['p_mw'] = net2.load['p_mw'] * 2.2
net2.load['q_mvar'] = net2.load['q_mvar'] * 2.2
pp.runpp(net2, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
max_loading_at_derate1 = net2.res_line['loading_percent'].max()
print(f"Max loading at 2.2x load, no derate: {max_loading_at_derate1:.6f}%")

# Calculate the derate needed to hit a target max loading of ~150%
target_loading = 150.0
calculated_derate = max_loading_at_derate1 / target_loading
print(f"\nCalculated derate for ~{target_loading}% target: {calculated_derate:.8f}")

print("\n=== Verifying with calculated derate ===")
net3 = pn.case89pegase()
net3.line['max_i_ka'] = net3.line['max_i_ka'] * calculated_derate
net3.load['p_mw'] = net3.load['p_mw'] * 2.2
net3.load['q_mvar'] = net3.load['q_mvar'] * 2.2
pp.runpp(net3, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
print(f"Max loading at 2.2x load, calculated derate: {net3.res_line['loading_percent'].max():.4f}%")