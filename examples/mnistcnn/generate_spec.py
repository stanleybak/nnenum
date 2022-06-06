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
from nnenum.network import nn_flatten

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

    with open("image0.vnnlib", 'w') as f:

        for image_id, (image, classification) in enumerate(zip(images, labels)):
            output = nn.execute(image)
            flat_output = nn_flatten(output)

            num_outputs = flat_output.shape[0]
            label = np.argmax(flat_output)

            if label == labels[image_id]:
                # correctly classified

                min_image = min_images[image_id]
                max_image = max_images[image_id]

                init_box = make_init_box(min_image, max_image)

                f.write(f"; GNN benchmark image {image_id + 1} with epsilon = {epsilon}\n\n")

                for i in range(len(init_box)):
                    f.write(f"(declare-const X_{i} Real)\n")

                f.write("\n")
                
                for i in range(10):
                    f.write(f"(declare-const Y_{i} Real)\n")

                f.write("\n; Input constraints:\n")

                for i, (lb, ub) in enumerate(init_box):
                    f.write(f"(assert (<= X_{i} {ub:.18f}))\n")
                    f.write(f"(assert (>= X_{i} {lb:.18f}))\n\n")

                f.write("\n; Output constraints:\n")
                f.write("(assert (or\n")

                for i in range(num_outputs):
                    if i == classification:
                        continue

                    f.write(f"    (and (>= Y_{i} Y_{classification}))\n")

                f.write("))")

                break

    return rv

def main():
    'main entry point'

    netname = 'mnist_0.1'
    onnx_filename = f'{netname}_noscale.onnx'
    epsilon = 0.1

    image_filename = 'mnist_test.csv'

    onnx_filename = f'{onnx_filename}'

    #nn = load_onnx_network(onnx_filename)
    #print(f"loading onnx network from {onnx_filename}")
    nn = load_onnx_network(onnx_filename)
    print(f"loaded network with {nn.num_relu_layers()} ReLU layers and {nn.num_relu_neurons()} ReLU neurons")

    specific_image = None
    print("Loading images...")
    tup_list = make_init(nn, image_filename, epsilon, specific_image=specific_image)
    print(f"made {len(tup_list)} init states")


if __name__ == '__main__':
    main()
 
