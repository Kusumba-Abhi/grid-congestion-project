import pandapower as pp
import pandapower.networks as pn

net = pn.case300()

print("=== case300 basic info ===")
print(f"Buses: {len(net.bus)}")
print(f"Lines: {len(net.line)}")
print(f"Generators: {len(net.gen)}")
print(f"Loads: {len(net.load)}")

print("\n=== Finding max viable load multiplier ===")
for multiplier in [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.2]:
    net_test = pn.case300()
    net_test.load['p_mw'] = net_test.load['p_mw'] * multiplier
    net_test.load['q_mvar'] = net_test.load['q_mvar'] * multiplier
    try:
        pp.runpp(net_test, enforce_q_lims=False, max_iteration=15, numba=False, init="flat")
        max_loading = net_test.res_line['loading_percent'].max()
        print(f"multiplier={multiplier}: CONVERGED, max_loading={max_loading:.4f}%")
    except Exception as e:
        print(f"multiplier={multiplier}: FAILED TO CONVERGE")