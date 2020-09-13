# nnenum - The Neural Network Enumeration Tool
**nnenum** (pronounced *en-en-en-um*) is a high-performance neural network verification tool. Multiple levels of abstraction are used to quickly verify ReLU networks without sacrificing completeness. Analysis combines three types of zonotopes with star set (triangle) overapproximations, and uses [efficient parallelized ReLU case splitting](http://stanleybak.com/papers/bak2020cav.pdf). The verification tree search can be augmented with adversarial example generation using multiple attacks from the [foolbox library](https://github.com/bethgelab/foolbox) to quickly find property violations. The tool is written in Python 3, uses GLPK for LP solving and directly accepts [ONNX](https://github.com/onnx/onnx) network files as input. The [ImageStar trick](https://arxiv.org/abs/2004.05511) allows sets to be quickly propagated through all layers supported by the [ONNX runtime](https://github.com/microsoft/onnxruntime), such as convolutional layers with arbitrary parameters.

The tool is written by Stanley Bak ([homepage](http://stanleybak.com), [twitter](https://twitter.com/StanleyBak)).

### Getting Started
The `Dockerfile` shows how to install all the dependencies (mostly python packages) and set up the environment. Although the tool loads neural networks directly from ONNX files, the properties and initial sets and verification settings must be defined in python scripts.

The best way to get started is to look at some of the examples. For example, in the `examples/acasxu` directory you can try to verify property 9 of network 3-3 of the [well-studied ACAS Xu neural network verification benchmarks](https://arxiv.org/abs/1702.01135) by running the command: 

```python3 acasxu_single.py 3 3 9```

### VNN 2020 Neural Network Verification Competition (VNN-COMP) Version
The nnenum tool performed well in VNN-COMP 2020, being the only tool to verify all the ACAS-Xu benchmarks (each in under 10 seconds), as well as one of the best on the MNIST and CIFAR-10 benchmarks. The version used for the competition as well as model files and scripts to run the compeition benchmarks are in the `vnn2020` branch.

### CAV 2020 Paper Version
The CAV 2020 paper ["Improved Geometric Path Enumeration for Verifying ReLU Neural Networks"](http://stanleybak.com/papers/bak2020cav.pdf) by S. Bak, H.D Tran, K. Hobbs and T. T. Johnson corresponds to optimizations integrated into the exact analysis mode of nnenum, which also benefits overapproximative analysis. The paper version and repeatability evaluation package instructions are available [here](http://stanleybak.com/papers/bak2020cav_repeatability.zip).
