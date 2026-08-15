import pandapower as pp
import pandapower.networks as pn

net = pn.case118()
print(f"Buses: {len(net.bus)}, Lines: {len(net.line)}, Generators: {len(net.gen)}")

pp.runpp(net)
print("Power flow converged successfully.")
print(net.res_line[['loading_percent']].head())