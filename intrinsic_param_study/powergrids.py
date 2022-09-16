import pandapower.networks as pn, pandapower as pp, pandas as pd, numpy as np

def two_bus_network():
    """
    panda_four_load_branch with 1 load.
    """
    net = pp.create_empty_network()

    #Buses
    bus1 = pp.create_bus(net, name="bus1", vn_kv=.4)
    bus2 = pp.create_bus(net, name="bus2", vn_kv=.4)

    #Generation (external) and load
    pp.create_ext_grid(net, bus1, name="External grid") # necessary for power flow
    pp.create_sgen(net, bus1, 0.12)
    pp.create_load(net, bus2, 0.12, 0.04)

    # Branch elements
    pp.create_line(net, bus1, bus2, name="line1", length_km=0.05, std_type="NAYY 4x120 SE")
    
    # Position elements
    n = bus2 + 1
    net.bus_geodata = pd.DataFrame(np.array([[0]*n, range(0, -n, -1)]).T, columns=['x', 'y'])
    return net
    
def n_bus_network():
    net = pn.create_cigre_network_mv(with_der="pv_wind")
    net.sgen.in_service.at[8] = False # only solar
    return net
