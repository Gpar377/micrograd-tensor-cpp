import numpy as np
import time
import torch
import sys
import os

# Ensure local autograd package is imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from autograd import Tensor

def benchmark_transformer_layer():
    # Model dimensions
    batch_size = 32
    seq_len = 128
    d_model = 512
    
    print("--- Running Autograd Transformer Layer Microbenchmarks ---")
    print(f"Batch Size: {batch_size} | Seq Len: {seq_len} | d_model: {d_model}")
    
    # Generate mock inputs (Queries, Keys, Values)
    Q_val = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
    K_val = np.random.randn(d_model, d_model).astype(np.float32)
    V_val = np.random.randn(d_model, d_model).astype(np.float32)
    
    # ------------------ Benchmark Custom Autograd Engine ------------------
    # Warmup
    Q = Tensor(Q_val, requires_grad=True)
    K = Tensor(K_val, requires_grad=True)
    V = Tensor(V_val, requires_grad=True)
    _ = (Q @ K) @ V
    
    start = time.perf_counter()
    for _ in range(5):
        # Forward pass
        attn_out = (Q @ K).relu() @ V
        loss = attn_out.sum()
        
        # Backward pass
        Q.grad, K.grad, V.grad = None, None, None
        loss.backward()
    end = time.perf_counter()
    t_custom = ((end - start) / 5) * 1000
    print(f"Custom Autograd Engine time: {t_custom:.2f} ms")

    # ------------------ Benchmark PyTorch CPU ------------------
    # Warmup
    Q_pt = torch.tensor(Q_val, requires_grad=True)
    K_pt = torch.tensor(K_val, requires_grad=True)
    V_pt = torch.tensor(V_val, requires_grad=True)
    _ = (Q_pt @ K_pt) @ V_pt
    
    start = time.perf_counter()
    for _ in range(5):
        # Forward pass
        attn_out_pt = torch.relu(Q_pt @ K_pt) @ V_pt
        loss_pt = attn_out_pt.sum()
        
        # Backward pass
        if Q_pt.grad is not None:
            Q_pt.grad.zero_()
            K_pt.grad.zero_()
            V_pt.grad.zero_()
        loss_pt.backward()
    end = time.perf_counter()
    t_torch = ((end - start) / 5) * 1000
    print(f"PyTorch CPU time:           {t_torch:.2f} ms")

    ratio = t_custom / t_torch
    print(f"Performance Ratio: {ratio:.2f}x of PyTorch CPU speed")
    
    # Print correctness confirmation
    assert np.allclose(attn_out.data, attn_out_pt.detach().numpy(), atol=1e-3)
    print("SUCCESS: Custom autograd outputs match PyTorch reference values!")

if __name__ == "__main__":
    benchmark_transformer_layer()
