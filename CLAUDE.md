# Claude Code Guidelines - Micrograd Tensor C++

## Project Overview
This repository contains a lightweight, reverse-mode automatic differentiation engine with C++ accelerated hot-paths.

## Technology Stack
*   **Python 3.10+** (Core Autograd framework, NumPy for validation)
*   **C++17** (PyBind11 performance extensions)
*   **Build Tools:** CMake / Setuptools (using `pybind11_add_module`)

## Coding Standards & Conventions
*   The computation graph must be represented as a Directed Acyclic Graph (DAG) with explicit topological sorting.
*   Every node in the graph must implement a `.backward()` hook that maps incoming gradients to inputs.
*   Write modular extensions: compile performance-critical loops (such as tensor contraction/matmul and activation functions) in C++.
*   Ensure broadcasting rules matches NumPy/PyTorch rules (e.g., standard dimension expansion and reduction of accumulated gradients).
*   Add exhaustive unit tests for every backward pass, verifying computed gradients against PyTorch gradients.

## Workflow Rules & Commands
*   **Install in Editable Mode:** `pip install -e .` (compiles C++ bindings)
*   **Run Unit Tests:** `pytest tests/`
*   **Train Verification Model:** `python scripts/train_mnist.py`
