'''
eth vnn benchmark 2020
'''

import sys
import time
from pathlib import Path

import onnx
import onnxruntime as ort

import numpy as np

from nnenum.onnx_network import load_onnx_network
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.lp_star_state import LpStarState
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.network import nn_flatten, nn_unflatten

def load_unscaled_images(filename, specific_image=None, epsilon=0.0):
    '''read images from csv file

    if epsilon is set, it gets added to the loaded image to get min/max images
    '''

    image_list = []
    labels = []

    line_num = 0

    mnist = 'mnist' in filename

    with open(filename, 'r') as f:
        line = f.readline()
                    
        while line is not None and len(line) > 0:
            line_num += 1

            if specific_image is not None and line_num - 1 != specific_image:
                line = f.readline()
                continue
            
            parts = line.split(',')
            labels.append(int(parts[0]))

            if mnist:
                line_list = [int(x.strip())/255.0 for x in parts[1:]]

                # add epsilon
                for i, val in enumerate(line_list):
                    line_list[i] = max(0.0, min(1.0, val + epsilon))
                
                image = np.array(line_list, dtype=np.float32)

                image.shape = (1, 1, 28, 28)
            else:
                #cifar load

                rgb_lists = [[], [], []]
                rgb_index = 0

                for x in parts[1:]:
                    val = int(x.strip())/255.0
                    val = max(0.0, min(1.0, val + epsilon))

                    rgb_lists[rgb_index].append(val)
                    rgb_index = (rgb_index + 1) % 3

                image = np.array(rgb_lists, dtype=np.float32)

                image.shape = (1, 3, 32, 32)

            image_list.append(image)

            line = f.readline()

    return image_list, labels

def make_init_box(min_image, max_image):
    'make init box'

    flat_min_image = nn_flatten(min_image)
    flat_max_image = nn_flatten(max_image)

    assert flat_min_image.size == flat_max_image.size

    box = list(zip(flat_min_image, flat_max_image))
        
    return box

def make_init(nn, image_filename, epsilon, specific_image=None):
    'returns list of (image_id, image_data, classification_label, init_star_state, spec)'

    rv = []

    images, labels = load_unscaled_images(image_filename, specific_image=specific_image)
    min_images, _ = load_unscaled_images(image_filename, specific_image=specific_image, epsilon=-epsilon)
    max_images, _ = load_unscaled_images(image_filename, specific_image=specific_image, epsilon=epsilon)

    print("making init states")
    
    for image_id, (image, classification) in enumerate(zip(images, labels)):
        output = nn.execute(image)
        flat_output = nn_flatten(output)

        num_outputs = flat_output.shape[0]
        label = np.argmax(flat_output)

        if label == labels[image_id]:
            # correctly classified

            # unsafe if classification is not maximal (anything else is > classfication)
            spec_list = []

            for i in range(num_outputs):
                if i == classification:
                    continue

                l = [0] * 10

                l[classification] = 1
                l[i] = -1

                spec_list.append(Specification([l], [0]))

            spec = DisjunctiveSpec(spec_list)

            min_image = min_images[image_id]
            max_image = max_images[image_id]

            init_box = make_init_box(min_image, max_image)
            init_box = np.array(init_box, dtype=np.float32)
            init_state = LpStarState(init_box, spec)

            image_index = image_id

            if specific_image is not None:
                image_index = specific_image

            rv.append((image_index, image, label, init_state, spec))

    return rv

def main():
    'main entry point'

    assert len(sys.argv) == 3, f"expected 2 args got {len(sys.argv)}: [network_name: 'mnist_0.1'] [timeout_mins (15)]"

    netname = sys.argv[1]
    onnx_filename = f'{netname}_noscale.onnx'
    epsilon = None

    if 'mnist' in netname:
        image_filename = 'data/mnist_test.csv'
    else:
        image_filename = 'data/cifar10_test.csv'

    if '0.1' in netname:
        epsilon = 0.1
    elif '0.3' in netname:
        epsilon = 0.3
    elif '2_255' in netname:
        epsilon = 2.0 / 255.0
    else:
        assert '8_255' in netname
        epsilon = 8.0 / 255.0

    timeout_mins = float(sys.argv[2])

    print(f"Using a timeout of {round(timeout_mins * 60, 1)} seconds")

    onnx_filename = f'data/{onnx_filename}'
    benchmark_name = f"eth_{netname}"

    filename = f'results/summary_{benchmark_name}.dat'

    Settings.TIMEOUT = 60 * timeout_mins
    Settings.COMPRESS_INIT_BOX = False
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    
    Settings.OVERAPPROX_MIN_GEN_LIMIT = np.inf
    Settings.SPLIT_IF_IDLE = False
    Settings.OVERAPPROX_LP_TIMEOUT = np.inf
    Settings.TIMING_STATS = False
    Settings.PRINT_OUTPUT = True

    Settings.ADVERSARIAL_TRY_QUICK = True

    # disable quick overapprox / adversarial on bigger networks
    if 'mnist_0.1' not in onnx_filename:
        Settings.TRY_QUICK_OVERAPPROX = False

    # contraction doesn't help in high dimensions
    Settings.OVERAPPROX_CONTRACT_ZONO_LP = False
    Settings.CONTRACT_ZONOTOPE = False
    Settings.CONTRACT_ZONOTOPE_LP = False

    #nn = load_onnx_network(onnx_filename)
    print(f"loading onnx network from {onnx_filename}")
    nn = load_onnx_network(onnx_filename)
    print(f"loaded network with {nn.num_relu_layers()} ReLU layers and {nn.num_relu_neurons()} ReLU neurons")

    Settings.ADVERSARIAL_ONNX_PATH = onnx_filename # path to .onnx file with corresponidng .onnx.pb file
    Settings.ADVERSARIAL_EPSILON = epsilon
    Settings.ADVERSARIAL_SEED_ABSTRACT_VIO = True

    #if epsilon == 0.05:
        # speed up splitting near root
    #    Settings.OVERAPPROX_TYPES_NEAR_ROOT = [['zono.area'],
    #                                       ['zono.area', 'zono.ybloat', 'zono.interval', 'star.quick']]

    results = {}
    safe_count = 0
    unsafe_count = 0
    unknown_count = 0
    error_count = 0

    specific_image = None
    print("Loading images...")
    tup_list = make_init(nn, image_filename, epsilon, specific_image=specific_image)
    print(f"made {len(tup_list)} init states")
    assert len(tup_list) > 0

    if specific_image is not None:
        Settings.TIMING_STATS = True

    Path("./results").mkdir(parents=True, exist_ok=True)

    with open(filename, 'w') as f:

        start = time.perf_counter()

        for index, (image_id, image_data, classification, init_state, spec) in enumerate(tup_list):
            print(f"\n========\nChecking robustness of image {image_id} for " + \
                  f"output class {classification}, with ep={epsilon}")

            # probably don't want to do this with globals
            Settings.ADVERSARIAL_ORIG_IMAGE = image_data
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

            tup = res.progress_tuple
            progress = f"{tup[0]}/{tup[0] + tup[1]} ({round(tup[2] * 100, 4)}%)"
            print(f"progress: {progress}")
            f.write(f"{benchmark_name}\t{image_id+1}\t{r}\t{t}\t{progress}\n")
            f.flush()

            tup_list[index] = None # clear memory from last star-state and computation

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
