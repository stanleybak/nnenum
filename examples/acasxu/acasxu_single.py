'''
Stanley Bak

generic ACAS-Xu analyis script for onnx networks

Determines network and property from command-line parameters.
'''

import sys

import numpy as np

from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.onnx_network import load_onnx_network_optimized
from nnenum.specification import Specification, DisjunctiveSpec

def get_init_box(property_str):
    'get lb, ub lists for the given property'

    if property_str in ("1", "2"):
        init_lb = [55947.691, -3.141592, -3.141592, 1145, 0]
        init_ub = [60760, 3.141592, 3.141592, 1200, 60]
    elif property_str == "3":
        init_lb = [1500, -0.06, 3.1, 980, 960]
        init_ub = [1800, 0.06, 3.141592, 1200, 1200]
    elif property_str == "4":
        init_lb = [1500, -0.06, 0, 1000, 700]
        init_ub = [1800, 0.06, 0, 1200, 800]
    elif property_str == "5":
        init_lb = [250, 0.2, -3.141592, 100, 0]
        init_ub = [400, 0.4, -3.141592 + 0.005, 400, 400]
    elif property_str == "6.1":
        init_lb = [12000, 0.7, -3.141592, 100, 0]
        init_ub = [62000, 3.141592, -3.141592 + 0.005, 1200, 1200]
    elif property_str == "6.2":
        init_lb = [12000, -3.141592, -3.141592, 100, 0]
        init_ub = [62000, -0.7, -3.141592 + 0.005, 1200, 1200]
    elif property_str == "7":
        init_lb = [0, -3.141592, -3.141592, 100, 0]
        init_ub = [60760, 3.141592, 3.141592, 1200, 1200]
    elif property_str == "8":
        init_lb = [0, -3.141592, -0.1, 600, 600]
        init_ub = [60760, -0.75*3.141592, 0.1, 1200, 1200]
    elif property_str == "9":
        init_lb = [2000, -0.4, -3.141592, 100, 0]
        init_ub = [7000, -0.14, -3.141592 + 0.01, 150, 150]
    elif property_str == "10":
        init_lb = [36000, 0.7, -3.141592, 900, 600]
        init_ub = [60760, 3.141592, -3.141592 + 0.01, 1200, 1200]
    else:
        raise RuntimeError(f"init_box undefined for property {property_str}")

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

def is_minimal_spec(indices):
    'create a disjunctive spec that a specific set of outputs is minimal'

    if isinstance(indices, int):
        indices = [indices]

    spec_list = []

    for i in range(5):
        if i in indices: # index 3 is strong left
            continue

        mat = []
        rhs = []

        for index in indices:
            l = [0, 0, 0, 0, 0]
            l[index] = -1
            l[i] = 1
            
            mat.append(l)
            rhs.append(0)

        spec_list.append(Specification(mat, rhs))

    return DisjunctiveSpec(spec_list)

def get_spec(property_str):
    'get the specification'

    #labels = ['Clear of Conflict (COC)', 'Weak Left', 'Weak Right', 'Strong Left', 'Strong Right']

    if property_str == "1":
        # unsafe if COC >= 1500

        # Output scaling is 373.94992 with a bias of 7.518884
        output_scaling_mean = 7.5188840201005975
        output_scaling_range = 373.94992
        
        # (1500 - 7.518884) / 373.94992 = 3.991125
        threshold = (1500 - output_scaling_mean) / output_scaling_range
        spec = Specification([[-1, 0, 0, 0, 0]], [-threshold])
        
    elif property_str == "2":
        # unsafe if COC is maximal:
        # y0 > y1 and y0 > y1 and y0 > y2 and y0 > y3 and y0 > y4
        spec = Specification([[-1, 1, 0, 0, 0],
                              [-1, 0, 1, 0, 0],
                              [-1, 0, 0, 1, 0],
                              [-1, 0, 0, 0, 1]], [0, 0, 0, 0])
    elif property_str in ("3", "4"):
        # unsafe if COC is minimal score
        spec = Specification([[1, -1, 0, 0, 0],
                              [1, 0, -1, 0, 0],
                              [1, 0, 0, -1, 0],
                              [1, 0, 0, 0, -1]], [0, 0, 0, 0])
    elif property_str == "5":
        # strong right should be minimal
        spec = is_minimal_spec(4)
    elif property_str in ["6.1", "6.2", "10"]:
        # coc should be minimal
        spec = is_minimal_spec(0)
    elif property_str == "7":
        # unsafe if strong left is minimial or strong right is minimal
        spec_left = Specification([[-1, 0, 0, 1, 0],
                                   [0, -1, 0, 1, 0],
                                   [0, 0, -1, 1, 0]], [0, 0, 0])

        spec_right = Specification([[-1, 0, 0, 0, 1],
                                    [0, -1, 0, 0, 1],
                                    [0, 0, -1, 0, 1]], [0, 0, 0])
        
        spec = DisjunctiveSpec([spec_left, spec_right])
    elif property_str == "8":
        # weak left is minimal or COC is minimal
        spec = is_minimal_spec([0, 1])
    elif property_str == "9":
        # strong left should be minimal...
        spec = is_minimal_spec(3)
    else:
        raise RuntimeError(f"spec undefined for property {property_str}")

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
    onnx_filename = (f'data/ACASXU_run2a_{n1}_{n2}_batch_2000.onnx')
    network = load_onnx_network_optimized(onnx_filename)

    return init_box, network, spec

def verify_acasxu(net, spec_str):
    ''''verify a system, 

    returns result_str, runtime (secs)
    '''

    required_spec_net_list = [
        ["5", (1, 1)],
        ["6", (1, 1)],
#        ["6.1", (1, 1)],
#        ["6.2", (1, 1)],
        ["7", (1, 9)],
        ["8", (2, 9)],
        ["9", (3, 3)],
        ["10", (4, 5)]]

    for req_spec, tup in required_spec_net_list:
        if spec_str == req_spec:
            assert net == tup, f"spec {spec_str} should only be run on net {tup}"

    result_str = None
    runtime = None

    if spec_str == "6":
        # disjunctive spec
        print("Running first part of spec 6")
        
        init_box, network, spec = load_init_network(net, "6.1")
        init_box = np.array(init_box, dtype=float)

        res = enumerate_network(init_box, network, spec)
        result_str = res.result_str
        runtime = res.total_secs
        print(f"First part of spec 6 finished in {runtime}")

        if result_str == "safe":
            print("Running second part of spec 6")
            init_box, network, spec = load_init_network(net, "6.2")
            init_box = np.array(init_box, dtype=float)

            res = enumerate_network(init_box, network, spec)
            result_str = res.result_str
            runtime += res.total_secs
            print(f"Second part of spec 6 finished in {res.total_secs}")
    else:
        init_box, network, spec = load_init_network(net, spec_str)
        init_box = np.array(init_box, dtype=float)

        res = enumerate_network(init_box, network, spec)
        result_str = res.result_str
        runtime = res.total_secs

    return result_str, runtime

def main():
    'main entry point'

    # change default settings for improved speed with ACAS Xu
    Settings.SPLIT_IF_IDLE = False
    Settings.PARALLEL_ROOT_LP = False
    Settings.PRINT_OVERAPPROX_OUTPUT = False
        
    if len(sys.argv) < 4:
        print("expected at least 3 args: net1 net2 spec_num <num_cores>")
        sys.exit(1)

    net1 = int(sys.argv[1])
    net2 = int(sys.argv[2])
    spec_str = sys.argv[3]

    if spec_str == "7":
        # ego is better at finding deep counterexamples
        Settings.BRANCH_MODE = Settings.BRANCH_EGO
        Settings.NUM_PROCESSES = 10
    else:
        Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX

    if len(sys.argv) > 4:
        cores = int(sys.argv[4])
        
        Settings.NUM_PROCESSES = cores
        print(f"Override num cores: {cores}")

    print(f"\nRunning with network {net1}-{net2} and spec {spec_str}")

    result_str, runtime = verify_acasxu((net1, net2), spec_str)

    print(f"Result for {net1}-{net2} and spec {spec_str}: {result_str}. Total runtime: {round(runtime, 2)} sec")

if __name__ == "__main__":
    main()
