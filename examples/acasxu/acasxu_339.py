'''
Stanley Bak

ACASXu nnenum example for network 3-3 property 9
'''

import numpy as np

from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.specification import Specification, DisjunctiveSpec

def get_init_box(property_str):
    'get lb, ub lists for the given property'

    assert property_str == "9"

    init_lb = [2000, -0.4, -3.141592, 100, 0]
    init_ub = [7000, -0.14, -3.141592 + 0.01, 150, 150]

    # scaling inputs
    means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
    range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

    num_inputs = len(init_lb)

    # normalize input
    for i in range(num_inputs):
        #print(f"unscaled Input {i}: {init_lb[i], init_ub[i]}")
        init_lb[i] = (init_lb[i] - means_for_scaling[i]) / range_for_scaling[i]
        init_ub[i] = (init_ub[i] - means_for_scaling[i]) / range_for_scaling[i]
        #print(f"scaled Input {i}: {init_lb[i], init_ub[i]}")

    return init_lb, init_ub

def is_minimal_spec(index):
    'create a disjunctive spec that a specific output is minimal'

    spec_list = []

    for i in range(5):
        if i == index: # index 3 is strong left
            continue

        l = [0, 0, 0, 0, 0]
        l[index] = -1
        l[i] = 1

        spec_list.append(Specification([l], [0]))

    return DisjunctiveSpec(spec_list)

def get_spec(property_str):
    'get the specification'

    assert property_str == "9"

    # strong left should be minimal...
    spec = is_minimal_spec(3)

    return spec

def load_init_network(net_pair, property_str):
    '''load the network / spec and return it

    the network is based on the net_pair, using specification spec

    returns (init_box, network, spec)
    '''

    # load the network and prepare input / output specs
    n1, n2 = net_pair
    
    init_lb, init_ub = get_init_box(property_str)
    init_box = list(zip(init_lb, init_ub))
        
    spec = get_spec(property_str)
    
    #network = weights_biases_to_nn(weights, biases)
    onnx_filename = (f'ACASXU_run2a_{n1}_{n2}_batch_2000.onnx')
    network = load_onnx_network_optimized(onnx_filename)

    return init_box, network, spec

def verify_acasxu(net, spec_str):
    ''''verify a system, 

    returns a verification result object'''

    init_box, network, spec = load_init_network(net, spec_str)

    init_box = np.array(init_box, dtype=float)

    res = enumerate_network(init_box, network, spec)

    return res

def main():
    'main entry point'

    # change default settings for improved speed with ACAS Xu
    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.OVERAPPROX_CONTRACT_ZONO_LP = False

    net1 = "3"
    net2 = "3"
    spec_str = "9"

    print(f"\nRunning with network {net1}-{net2} and spec {spec_str}")
    res = verify_acasxu((net1, net2), spec_str)
    print(f"Result: {res.result_str} ({round(res.total_secs, 2)} sec)")

if __name__ == "__main__":
    main()
