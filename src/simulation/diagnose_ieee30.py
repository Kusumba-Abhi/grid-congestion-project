import pandapower as pp
import pandapower.networks as pn

net = pn.case_ieee30()

print("=== Line thermal ratings (max_i_ka) BEFORE any derating ===")
print(net.line[['max_i_ka']])

print("\n=== Running power flow with NO derating, NO extra load/renewables ===")
pp.runpp(net, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
print(net.res_line[['loading_percent']])

print(f"\nMax loading at baseline: {net.res_line['loading_percent'].max():.4f}%")

print("\n=== Now with load multiplier x2.2 (top of our current range) ===")
net2 = pn.case_ieee30()
net2.load['p_mw'] = net2.load['p_mw'] * 2.2
net2.load['q_mvar'] = net2.load['q_mvar'] * 2.2
pp.runpp(net2, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
print(f"Max loading at 2.2x load, no derate: {net2.res_line['loading_percent'].max():.4f}%")

print("\n=== Now with load multiplier x2.2 AND thermal_derate 0.01 ===")
net3 = pn.case_ieee30()
net3.line['max_i_ka'] = net3.line['max_i_ka'] * 0.01
net3.load['p_mw'] = net3.load['p_mw'] * 2.2
net3.load['q_mvar'] = net3.load['q_mvar'] * 2.2
pp.runpp(net3, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
print(f"Max loading at 2.2x load, 0.01 derate: {net3.res_line['loading_percent'].max():.4f}%")