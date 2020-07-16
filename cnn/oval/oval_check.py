'''
oval vnn benchmark 2020
'''

import sys
import time
import pickle

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nnenum.onnx_network import load_onnx_network
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.lp_star_state import LpStarState
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.network import nn_flatten, nn_unflatten

def make_init_box(min_image, max_image):
    'make init box'

    flat_min_image = nn_flatten(min_image)
    flat_max_image = nn_flatten(max_image)

    assert flat_min_image.size == flat_max_image.size

    box = list(zip(flat_min_image, flat_max_image))
        
    return box

def make_init(nn, image_filename, specific_image=None, normalize=True):
    'returns list of (image_id, image_data, classification_label, init_star_state, spec)'

    if specific_image is not None:
        print(f"WARNING: using specific image = {specific_image}")
    else:
        print("Loading all image data...")

    with open(image_filename, 'rb') as f:
        data = pickle.load(f)

    if specific_image is not None:
        data = [data[specific_image]]

    print(f"Loading {len(data)} images")
    
    rv = []

    for cifar_index, epsilon, prop, cifar_label, im in data:
        print(".", end="", flush=True)

        #print("WARNING: USiNG REDUCED EPSILON VALUE")
        #epsilon *= 0.85

        # scale to [1.0, 0.0]
        im = im / 255.0
        im = im.astype(np.float32)
        im.shape = (1, 3, 32, 32)

        assert np.min(im) >= 0.0
        assert np.max(im) <= 1.0, f"max was {np.max(im)}"

        if not normalize:
            print("WARNING: normalize = False")

        # normalize before scaling
        #min_image = np.clip(im.copy() - epsilon, 0, 1)
        #max_image = np.clip(im.copy() + epsilon, 0, 1)

        if normalize:
            # apply normalization
            mean = [0.485, 0.456, 0.406]
            std = [0.225, 0.225, 0.225]

            # normalize before scaling
            #for channel in range(3):
            #    for idata in [min_image, max_image]:
            #        idata[0, channel] = (idata[0, channel] - mean[channel]) / std[channel]

            for channel in range(3):
                im[0, channel] = (im[0, channel] - mean[channel]) / std[channel]

        if not normalize:
            # scale epsilon
            epsilon *= 0.225 # std is 0.225 for all channels

        # normalize AFTER scaling
        min_image = im.copy() - epsilon
        max_image = im.copy() + epsilon

        if not normalize:
            min_image = np.clip(min_image, 0, 1)
            max_image = np.clip(max_image, 0, 1)

        if normalize:
            # clip at limits
            for channel in range(3):
                ub = (1.0 - mean[channel]) / std[channel]
                lb = (0.0 - mean[channel]) / std[channel]

                min_image[0, channel] = np.clip(min_image[0, channel], lb, ub)
                max_image[0, channel] = np.clip(max_image[0, channel], lb, ub)

        assert np.min(np.abs(min_image - max_image)) > 1e-5, "inf-norm perturbed images should be completely different"
        
        assert nn.get_input_shape() == im.shape

        output = np.ravel(nn.execute(im))
        label = np.argmax(output)

        #print(f"original output (class {label}): {output}")
        
        assert label == cifar_label, f"{len(rv)}: cifar label incorrect for cifar_index {cifar_index}, " + \
            f"got {label} expected {cifar_label}"

        mat = []

        #for index in range(10):
        #    if index == prop:
        #        continue
            
        #    l = [0.0] * 10
        #    l[index] = 1
        #    l[prop] = -1

        #    mat.append(l)
        
        l = [0] * 10

        l[label] = 1
        l[prop] = -1
        mat.append(l)

        spec = Specification(mat, [0] * len(mat))

        init_box = make_init_box(min_image, max_image)
        
        init_box = np.array(init_box, dtype=np.float32)

        init_state = LpStarState(init_box, spec)
        
        if specific_image is not None:
            image_id = specific_image
        else:
            image_id = len(rv)

        rv.append((image_id, im, label, init_state, epsilon, spec, prop))

    print('')

    return rv

def main():
    'main entry point'

    assert len(sys.argv) == 3, f"expected 2 args got {len(sys.argv) - 1}: [network_name: 'base'] [timeout_mins (15)]"

    netname = sys.argv[1]

    assert netname in ['base', 'deep', 'wide'], f"unknown netname: {netname}"
    
    onnx_filename = f'data/cifar_{netname}_kw.onnx'
    image_filename = f'data/{netname}_data.pkl'

    normalize = 'noscale' not in onnx_filename

    timeout_mins = float(sys.argv[2])

    print(f"Using a timeout of {round(timeout_mins * 60, 1)} seconds")

    benchmark_name = f"oval_{netname}100"

    filename = f'results/summary_{benchmark_name}.dat'

    Settings.TIMEOUT = 60 * timeout_mins
    Settings.COMPRESS_INIT_BOX = False
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    
    Settings.OVERAPPROX_MIN_GEN_LIMIT = np.inf
    Settings.SPLIT_IF_IDLE = False
    Settings.OVERAPPROX_LP_TIMEOUT = np.inf
    Settings.TIMING_STATS = True
    Settings.PRINT_OUTPUT = True

    # using dual glpk here given glpk errors (ill-conditioned bm, fail with primary settings)
    # no resets is faster than with resets
    Settings.ADVERSARIAL_TEST_ABSTRACT_VIO = False

    #Settings.NUM_PROCESSES = 1

    # disable quick adversarial on bigger networks
    Settings.ADVERSARIAL_TRY_QUICK = False
    Settings.TRY_QUICK_OVERAPPROX = False

    # contraction doesn't help in high dimensions
    Settings.OVERAPPROX_CONTRACT_ZONO_LP = False
    Settings.CONTRACT_ZONOTOPE = False
    Settings.CONTRACT_ZONOTOPE_LP = False

    #nn = load_onnx_network(onnx_filename)
    print(f"loading onnx network from {onnx_filename}")
    nn = load_onnx_network(onnx_filename)
    print(f"loaded network with {nn.num_relu_layers()} ReLU layers and {nn.num_relu_neurons()} ReLU neurons")

    if not normalize:
        Settings.ADVERSARIAL_ONNX_PATH = onnx_filename # path to .onnx file with corresponidng .onnx.pb file
        Settings.ADVERSARIAL_EPSILON = None # populated on a per-benchmark basis
        Settings.ADVERSARIAL_SEED_ABSTRACT_VIO = False # true

    #if epsilon == 0.05:
        # speed up splitting near root
    #print("using quick overapprox near root")
    #Settings.OVERAPPROX_TYPES_NEAR_ROOT = [['zono.area']]
                                           #['zono.area', 'zono.ybloat', 'zono.interval', 'star.quick']]

    results = {}
    safe_count = 0
    unsafe_count = 0
    unknown_count = 0
    error_count = 0

    specific_image = 45 #None #38

    tup_list = make_init(nn, image_filename, specific_image=specific_image, normalize=normalize)
    print(f"made {len(tup_list)} init states")
    assert len(tup_list) > 0

    if specific_image is not None:
        Settings.TIMING_STATS = True

    #Settings.QUICK_OVERAPPROX_TYPES = [['zono.area'],
    #                                   ['zono.area', 'zono.ybloat', 'zono.interval'],
    #                                   ['zono.area', 'zono.ybloat', 'zono.interval', 'star.quick']]

    if Settings.NUM_PROCESSES == 1:
        print("WARNING using single process")

    Path("./results").mkdir(parents=True, exist_ok=True)

    with open(filename, 'w') as f:

        start = time.perf_counter()

        for image_id, image_data, label, init_state, epsilon, spec, target in tup_list:
            print(f"\n========\nChecking robustness of image {image_id} for " + \
                  f"output class {label}->{target}, with ep={epsilon}")

            Settings.ADVERSARIAL_EPSILON = epsilon
            Settings.ADVERSARIAL_ORIG_IMAGE = image_data
            Settings.ADVERSARIAL_ORIG_LABEL = label
            Settings.ADVERSARIAL_TARGET = target
            Settings.ADVERSARIAL_SEED_ABSTRACT_VIO = True

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

            tup = res.progress_tuple
            progress = f"{tup[0]}/{tup[0] + tup[1]} ({round(tup[2] * 100, 4)}%)"
            print(f"progress: {progress}")
            f.write(f"{benchmark_name}\t{image_id}\t{r}\t{t}\t{progress}\n")
            f.flush()

        diff = time.perf_counter() - start
        f.write(f"\nSafe: {safe_count}, unsafe: {unsafe_count}, error: {error_count}, unknown: {unknown_count}, " + \
                f"Total Time: {round(diff, 2)} sec\n")

        with open(f'results/accumulated_{benchmark_name}.dat', 'w') as f2:
            for index, v in enumerate(sorted(results.values())):
                t = v[0]
                f2.write(f"{t}\t{index+1}\n")

        for v in sorted(results.values()):
            print(v)
                
        print(f"safe: {safe_count}, unsafe: {unsafe_count}, unknown: {unknown_count}")

if __name__ == '__main__':
    main()
