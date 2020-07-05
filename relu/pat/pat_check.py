'''
pat vnn benchmark 2020
'''

import sys
import time
from pathlib import Path

import numpy as np

from nnenum.onnx_network import load_onnx_network, load_onnx_network_optimized
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.lp_star_state import LpStarState
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings

def load_image(num):
    'test reading images from csv file'

    image_data = []

    filename = f'data/image{num}'

    with open(filename, 'r') as f:
        line = f.readline()

        while line is not None and len(line) > 0:
            parts = line.split(',')

            # there's a trailing comma
            line_list = [float(x.strip())/255.0 for x in parts[:-1]]

            image_data += line_list

            line = f.readline()

    np_image_data = np.array(image_data, dtype=np.dtype('float32'))
    np_image_data.shape = (1, 784, 1)

    return np_image_data

def make_init_box(image, epsilon):
    'make init'

    box = []

    #init_lb = []
    #init_ub = []

    for x in np.ravel(image):
        lb = max(0, x - epsilon)
        ub = min(1.0, x + epsilon)
        
        #init_lb.append(lb)
        #init_ub.append(ub)
        box.append((lb, ub))
        
    #return np.array(list(zip(init_lb, init_ub)), dtype=np.float32)
    return box

def main():
    'main entry point'

    #epsilon = 0.05
    #onnx_filename = 'mnist-net_256x6.onnx'

    assert len(sys.argv) == 4, f"expected 3 args got {len(sys.argv)}: [network (2/4/6)] [epsilon (0.02)] [timeout_mins (15)]"

    net = sys.argv[1]
    assert net in ["2", "4", "6"]

    epsilon = float(sys.argv[2])
    assert 0 <= epsilon < 0.1, f"unusual value of epsilon (did you miss a 0?): {epsilon}"

    timeout_mins = float(sys.argv[3])

    onnx_filename = f'data/mnist-net_256x{net}.onnx'

    name = f"pat_net{net}_ep{epsilon}"

    filename = f'results/summary_{name}.dat'

    Settings.TIMEOUT = 60 * timeout_mins
    Settings.COMPRESS_INIT_BOX = False
    
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    
    Settings.OVERAPPROX_MIN_GEN_LIMIT = np.inf
    Settings.SPLIT_IF_IDLE = False
    Settings.OVERAPPROX_LP_TIMEOUT = np.inf
    Settings.TIMING_STATS = False
    Settings.PRINT_OUTPUT = True

    # contraction doesn't help in high dimensions
    Settings.OVERAPPROX_CONTRACT_ZONO_LP = False
    Settings.CONTRACT_ZONOTOPE = False
    Settings.CONTRACT_ZONOTOPE_LP = False

    #nn = load_onnx_network(onnx_filename)
    nn = load_onnx_network_optimized(onnx_filename)

    Settings.ADVERSARIAL_ONNX_PATH = onnx_filename # path to .onnx file with corresponidng .onnx.pb file
    Settings.ADVERSARIAL_EPSILON = epsilon

    Settings.ADVERSARIAL_FROM_ABSTRACT_VIO = True

    if epsilon == 0.05:
        # speed up splitting near root
        Settings.OVERAPPROX_TYPES_NEAR_ROOT = [['zono.area'],
                                           ['zono.area', 'zono.ybloat', 'zono.interval', 'star.quick']]

    results = {}
    safe_count = 0
    unsafe_count = 0
    unknown_count = 0
    error_count = 0

    tup_list = [] # list of image_id, classification, init_state, spec

    print("Loading images...")

    for image_id in range(1, 26):
        image = load_image(image_id)
        out = nn.execute(image)
        classification = np.argmax(out)

        init_box = make_init_box(image, epsilon)

        # unsafe if classification is not maximal (anything else is > classfication)
        spec_list = []

        for i in range(10):
            if i == classification:
                continue

            l = [0] * 10

            l[classification] = 1
            l[i] = -1

            spec_list.append(Specification([l], [0]))

        spec = DisjunctiveSpec(spec_list)

        bounds = np.array(init_box, dtype=np.float32)
        init_state = LpStarState(bounds, spec)

        tup_list.append((image_id, classification, image, init_state, spec))

    Path("./results").mkdir(parents=True, exist_ok=True)

    with open(filename, 'w') as f:

        start = time.perf_counter()
        for image_id, classification, image, init_state, spec in tup_list:
            print(f"\n========\nChecking robustness of image {image_id} for output class {classification}, with ep={epsilon}")

            # probably don't want to do this with globals
            Settings.ADVERSARIAL_ORIG_IMAGE = image
            Settings.ADVERSARIAL_ORIG_LABEL = classification

            res = enumerate_network(init_state, nn, spec)
            r = res.result_str
            t = res.total_secs

            if r == 'safe':
                safe_count += 1
            elif r == 'unsafe':
                unsafe_count += 1
            else:
                if r == 'error':
                    error_count += 1
                    
                unknown_count += 1

            if r != "timeout":
                results[image_id] = [t, r, image_id]

            print(f"result {image_id}: {r} in {round(t, 4)} sec")

            f.write(f"net{net} (e={epsilon})\t{image_id}\t{r}\t{t}\n")
            f.flush()

        diff = time.perf_counter() - start
        f.write(f"\nSafe: {safe_count}, unsafe: {unsafe_count}, error: {error_count}, unknown: {unknown_count}, " + \
                f"Total Time: {round(diff, 2)} sec\n")

        with open(f'results/accumulated_{name}.dat', 'w') as f2:
            for index, v in enumerate(sorted(results.values())):
                t = v[0]
                f2.write(f"{t}\t{index+1}\n")

        for v in sorted(results.values()):
            print(v)
                
        print(f"safe: {safe_count}, unsafe: {unsafe_count}, unknown: {unknown_count}")

if __name__ == '__main__':
    main()
