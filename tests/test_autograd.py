import numpy as np
import torch
import pytest
from autograd import Tensor

def test_scalar_arithmetic():
    # Setup inputs
    x_val, y_val = 3.0, -4.0
    
    # Custom autograd
    x = Tensor(x_val, requires_grad=True)
    y = Tensor(y_val, requires_grad=True)
    z = x * y + y
    z.backward()
    
    # PyTorch reference
    x_pt = torch.tensor(x_val, requires_grad=True)
    y_pt = torch.tensor(y_val, requires_grad=True)
    z_pt = x_pt * y_pt + y_pt
    z_pt.backward()
    
    assert np.allclose(z.data, z_pt.detach().numpy())
    assert np.allclose(x.grad, x_pt.grad.numpy())
    assert np.allclose(y.grad, y_pt.grad.numpy())

def test_matrix_multiplication():
    # Shape 3x4 and 4x5
    A_val = np.random.randn(3, 4).astype(np.float32)
    B_val = np.random.randn(4, 5).astype(np.float32)
    
    # Custom
    A = Tensor(A_val, requires_grad=True)
    B = Tensor(B_val, requires_grad=True)
    C = A @ B
    
    # Simulate a downstream gradient
    grad_output = np.random.randn(3, 5).astype(np.float32)
    C.backward(grad_output)
    
    # PyTorch
    A_pt = torch.tensor(A_val, requires_grad=True)
    B_pt = torch.tensor(B_val, requires_grad=True)
    C_pt = A_pt @ B_pt
    C_pt.backward(torch.tensor(grad_output))
    
    assert np.allclose(C.data, C_pt.detach().numpy(), atol=1e-4)
    assert np.allclose(A.grad, A_pt.grad.numpy(), atol=1e-4)
    assert np.allclose(B.grad, B_pt.grad.numpy(), atol=1e-4)

def test_broadcasting_addition():
    # Broadcating a 1D vector (bias) to a 2D matrix (activations)
    x_val = np.random.randn(5, 3).astype(np.float32)
    b_val = np.random.randn(1, 3).astype(np.float32)
    
    # Custom
    x = Tensor(x_val, requires_grad=True)
    b = Tensor(b_val, requires_grad=True)
    z = (x + b).relu()
    
    grad_output = np.random.randn(5, 3).astype(np.float32)
    z.backward(grad_output)
    
    # PyTorch
    x_pt = torch.tensor(x_val, requires_grad=True)
    b_pt = torch.tensor(b_val, requires_grad=True)
    z_pt = torch.relu(x_pt + b_pt)
    z_pt.backward(torch.tensor(grad_output))
    
    assert np.allclose(z.data, z_pt.detach().numpy(), atol=1e-4)
    assert np.allclose(x.grad, x_pt.grad.numpy(), atol=1e-4)
    assert np.allclose(b.grad, b_pt.grad.numpy(), atol=1e-4)

def test_reduction_sum():
    x_val = np.random.randn(4, 4).astype(np.float32)
    
    # Custom
    x = Tensor(x_val, requires_grad=True)
    y = x.sum(axis=0)
    
    grad_output = np.random.randn(4).astype(np.float32)
    y.backward(grad_output)
    
    # PyTorch
    x_pt = torch.tensor(x_val, requires_grad=True)
    y_pt = x_pt.sum(dim=0)
    y_pt.backward(torch.tensor(grad_output))
    
    assert np.allclose(y.data, y_pt.detach().numpy(), atol=1e-4)
    assert np.allclose(x.grad, x_pt.grad.numpy(), atol=1e-4)
