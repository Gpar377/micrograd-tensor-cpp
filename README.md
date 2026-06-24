# Tensor Autograd Engine with C++ Extensions

A reverse-mode automatic differentiation (autograd) engine built from scratch. Exposes dynamic DAG computation graphs, automatic gradient backpropagation with tensor shape broadcasting/unbroadcasting, and custom C++ extensions compiled and bound via PyBind11.

---

## 🛠️ Installation and Building

### 1. Build and install custom C++ Extensions
Builds the performance critical hot-paths (`autograd_backend.so`) directly in editable mode:
```bash
pip install -e .
```

### 2. Verify Correctness (Tests)
Executes our comprehensive mathematical validation suite checking calculated weights/gradients against PyTorch values:
```bash
pytest tests/
```

### 3. Run Microbenchmarks
Profiles execution speeds of the Custom C++ Backend vs. PyTorch CPU for a standard Transformer attention layer:
```bash
python benchmark.py
```

---

## 🏗️ Architecture Design

*   **DAG Construction:** Each computed `Tensor` tracks its parents and the operation that created it.
*   **Automatic Differentiation:** Reverse-mode backpropagation executing topological sorting over the computation graph.
*   **Broadcasting Engine:** Mathematical mapping to reduce and scale incoming gradients where input sizes were broadcasted.
*   **C++ Extension Layer (`src/tensor_ops.cpp`):** Exposes OpenMP parallelized matrix multiplications and ReLU activation loops to Python using PyBind11.
