# nnenum - The Neural Network Enumeration Tool
**nnenum** (pronounced *en-en-en-um*) is a neural network verification tool written in Python 3. It can be used to prove input / output properties about neural networks with ReLU activation functions, such as the absence of adversarial examples. The underlying algorithm is based on the linear star set (AH-Polytope) data structure, combined with partial overapproximation analysis, adversarial input generation and many optimizations.

For some examples of how to run nnenum, see the examples folder

# CAV 2020 Paper Version
The CAV 2020 paper "Improved Geometric Path Enumeration for Verifying ReLU Neural Networks" by S. Bak, H.D Tran, K. Hobbs and T. T. Johnson corresponds to optimizations for the exact analysis mode of nnenum. The paper version and repeatability evaluation package instructions are available here: http://stanleybak.com/papers/bak2020cav_repeatability.zip
