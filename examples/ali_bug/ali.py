'''
nnenum vnnlib front end

usage: "python3 nnenum.py <onnx_file> <vnnlib_file> [timeout=None] [outfile=None]"

Stanley Bak
June 2021
'''

import sys

import numpy as np

from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.result import Result
from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.vnnlib import get_num_inputs_outputs, read_vnnlib_simple

def make_spec(vnnlib_filename, onnx_filename):
    '''make Specification

    returns a pair: (list of [box, Specification], inp_dtype)
    '''

    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(onnx_filename)
    vnnlib_spec = read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs)

    rv = []

    for box, spec_list in vnnlib_spec:
        if len(spec_list) == 1:
            mat, rhs = spec_list[0]
            spec = Specification(mat, rhs)
        else:
            spec_obj_list = [Specification(mat, rhs) for mat, rhs in spec_list]
            spec = DisjunctiveSpec(spec_obj_list)

        rv.append((box, spec))

    return rv, inp_dtype

def set_control_settings():
    'set settings for smaller control benchmarks'

    Settings.TIMING_STATS = False
    Settings.PARALLEL_ROOT_LP = False
    Settings.SPLIT_IF_IDLE = False
    Settings.PRINT_OVERAPPROX_OUTPUT = False
    Settings.TRY_QUICK_OVERAPPROX = True

    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.CONTRACT_LP_OPTIMIZED = True
    Settings.CONTRACT_LP_TRACK_WITNESSES = True

    Settings.OVERAPPROX_BOTH_BOUNDS = False

    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    Settings.OVERAPPROX_GEN_LIMIT_MULTIPLIER = 1.5
    Settings.OVERAPPROX_LP_TIMEOUT = 0.02
    Settings.OVERAPPROX_MIN_GEN_LIMIT = 70

def set_exact_settings():
    'set settings for smaller control benchmarks'

    Settings.TIMING_STATS = True
    Settings.TRY_QUICK_OVERAPPROX = False

    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.CONTRACT_LP_OPTIMIZED = True
    Settings.CONTRACT_LP_TRACK_WITNESSES = True

    Settings.OVERAPPROX_BOTH_BOUNDS = False

    Settings.BRANCH_MODE = Settings.BRANCH_EXACT

def set_image_settings():
    'set settings for larger image benchmarks'

    Settings.COMPRESS_INIT_BOX = True
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    Settings.TRY_QUICK_OVERAPPROX = False
    
    Settings.OVERAPPROX_MIN_GEN_LIMIT = np.inf
    Settings.SPLIT_IF_IDLE = False
    Settings.OVERAPPROX_LP_TIMEOUT = np.inf
    Settings.TIMING_STATS = True

    # contraction doesn't help in high dimensions
    #Settings.OVERAPPROX_CONTRACT_ZONO_LP = False
    Settings.CONTRACT_ZONOTOPE = False
    Settings.CONTRACT_ZONOTOPE_LP = False

def main():
    'main entry point'

    onnx_filename = "mnist-net_256x4.onnx"
    vnnlib_filename = "prop_14_0.05.vnnlib"
    timeout = None
    outfile = None

    network = load_onnx_network_optimized(onnx_filename)

    spec_list, input_dtype = make_spec(vnnlib_filename, onnx_filename)
    assert len(spec_list) == 1
    init_box, spec = spec_list[0]
    init_box = np.array(init_box, dtype=input_dtype)

    set_image_settings()
    # set_control_settings()
    # set_exact_settings()

    Settings.OVERAPPROX_TYPES_NEAR_ROOT = Settings.OVERAPPROX_TYPES = [['zono.area'],
                                ['zono.area', 'zono.ybloat', 'zono.interval'],
                                ['zono.area', 'zono.ybloat', 'zono.interval', 'star.lp']]
    Settings.OVERAPPROX_MIN_GEN_LIMIT = np.inf
    Settings.OVERAPPROX_LP_TIMEOUT = np.inf
    Settings.PARALLEL_ROOT_LP = False
    Settings.NUM_PROCESSES = 1
    Settings.OVERAPPROX_BOTH_BOUNDS = True

    res = enumerate_network(init_box, network, spec)
    result_str = res.result_str

    # rename for VNNCOMP21:
        
    if result_str == "safe":
        result_str = "holds"
    elif "unsafe" in result_str:
        result_str = "violated"

    if outfile is not None:
        with open(outfile, 'w') as f:
            f.write(result_str)
            
    #print(result_str)

    if result_str == 'error':
        sys.exit(Result.results.index('error'))


if __name__ == '__main__':
    main()
