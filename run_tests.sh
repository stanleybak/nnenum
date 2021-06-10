#!/bin/bash -e

python3 -m nnenum.nnenum examples/acasxu/data/ACASXU_run2a_1_1_batch_2000.onnx examples/acasxu/data/prop_1.vnnlib


python3 -m nnenum.nnenum examples/mnistfc/mnist-net_256x2.onnx examples/mnistfc/prop_0_0.03.vnnlib

python3 -m nnenum.nnenum examples/mnistfc/mnist-net_256x2.onnx examples/mnistfc/prop_2_0.03.vnnlib

python3 -m nnenum.nnenum examples/cifar2020/cifar10_2_255_simplified.onnx examples/cifar2020/cifar10_spec_idx_11_eps_0.00784_n1.vnnlib
