import numpy as np
from .tensor import Tensor

class Op:
    """
    Base class for all mathematical operators in the computation graph.
    """
    @classmethod
    def apply(cls, *args):
        # Convert raw values/scalars to Tensors
        inputs = []
        for arg in args:
            if isinstance(arg, Tensor):
                inputs.append(arg)
            else:
                inputs.append(Tensor(arg))

        # Instantiate operation context
        ctx = cls()
        ctx.inputs = inputs

        # Check if any input requires gradient tracking
        requires_grad = any(t.requires_grad for t in inputs)

        # Execute forward pass
        raw_inputs = [t.data for t in inputs]
        out_data = ctx.forward(*raw_inputs)

        # Create output Tensor linked back to this operation node
        out = Tensor(out_data, requires_grad=requires_grad, _creator=ctx)
        return out

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


# ------------------- Operation Implementations -------------------

class Add(Op):
    def forward(self, x, y):
        return x + y

    def backward(self, grad_output):
        # Derivatives of (x + y) w.r.t x and y is 1.0
        return grad_output, grad_output


class Neg(Op):
    def forward(self, x):
        return -x

    def backward(self, grad_output):
        # Derivative of (-x) w.r.t x is -1.0
        return -grad_output


class Mul(Op):
    def forward(self, x, y):
        # Save inputs for backward gradient calculations
        self.x = x
        self.y = y
        return x * y

    def backward(self, grad_output):
        # Derivative of (x * y) is y (w.r.t x) and x (w.r.t y)
        grad_x = grad_output * self.y
        grad_y = grad_output * self.x
        return grad_x, grad_y


class MatMul(Op):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return np.matmul(x, y)

    def backward(self, grad_output):
        # Derivative of Matrix Multiplication
        # dL/dX = dL/dY * Y^T
        # dL/dY = X^T * dL/dY
        grad_x = np.matmul(grad_output, self.y.T)
        grad_y = np.matmul(self.x.T, grad_output)
        return grad_x, grad_y


class ReLU(Op):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad_output):
        # Derivative of ReLU is 1 if x > 0 else 0
        return grad_output * (self.x > 0)


class Sum(Op):
    def forward(self, x, axis=None, keepdims=False):
        self.x_shape = x.shape
        self.axis = axis
        self.keepdims = keepdims
        return np.sum(x, axis=axis, keepdims=keepdims)

    def backward(self, grad_output):
        # Derivative of sum: broadcast gradient vector back to original input shape
        if not self.keepdims and self.axis is not None:
            # Reshape grad_output to restore squeezed dimensions for correct broadcasting
            axes = [self.axis] if isinstance(self.axis, int) else self.axis
            shape = list(self.x_shape)
            for ax in axes:
                shape[ax] = 1
            grad_output = grad_output.reshape(shape)
            
        return np.broadcast_to(grad_output, self.x_shape)
