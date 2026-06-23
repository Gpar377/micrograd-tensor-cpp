# Tensor Autograd Engine
A lightweight, reverse-mode automatic differentiation (autograd) engine built from scratch. It features a dynamically constructed computation graph, support for tensor operations with broadcasting, multi-dimensional tensor reshapes, and a high-performance C++ backend extension for computationally heavy operations.

## Proposed Git Repo Name
`micrograd-tensor-cpp`

## Architecture & Scope
*   **Dynamic Computation Graph:** DAG construction where each Tensor tracks its inputs (creators) and the operation that generated it.
*   **Reverse-Mode Automatic Differentiation:** Topological sorting of the DAG followed by backward propagation of gradients using the chain rule.
*   **Tensor Math Support:** Matrix multiplication, element-wise ops (add, sub, mul, div), activation functions (ReLU, Sigmoid, Softmax), reduction ops (sum, mean along axes), and shape transformations (transpose, reshape, slice).
*   **Broadcasting Engine:** Automatic tracking and scaling of gradients when backpropagating through shape broadcasting (e.g., adding a batch tensor to a bias vector).
*   **C++ Execution Backend:** Offloading core numerical operations (matmul, matrix-vector ops) to C++ using PyBind11 to avoid Python interpreter overhead.
*   **Neural Network API:** Implementation of standard layers (`Linear`, `Conv2D`, `LayerNorm`, `Dropout`) and optimizers (`SGD`, `Adam`).
*   **Validation:** Training a 6-layer Transformer or multi-layer perceptron (MLP) on MNIST/Tiny Shakespeare and benchmarking execution speed and convergence against PyTorch.

## Target Milestones
1. Core Tensor class with dynamic DAG tracking and scalar auto-diff.
2. Tensor shape operations, broadcasting, and backward passes for tensor-tensor arithmetic.
3. PyBind11 C++ backend extension implementation.
4. Neural network layers (`nn.Module`, `nn.Linear`), loss functions, and `Adam` optimizer.
5. Transformer model training script and benchmarking suite.
