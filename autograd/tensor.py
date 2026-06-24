import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _creator=None):
        # Enforce data as NumPy array of float32 type
        if isinstance(data, (int, float)):
            data = [data]
        self.data = np.asarray(data, dtype=np.float32)
        
        self.requires_grad = requires_grad
        self.grad = None
        self._creator = _creator
        self._parents = []
        
        if _creator is not None:
            self._parents = _creator.inputs

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor({self.data.tolist()}, requires_grad={self.requires_grad})"

    def backward(self, grad=None):
        """
        Executes reverse-mode automatic differentiation.
        Uses topological sorting to traverse the computation graph DAG.
        """
        if not self.requires_grad:
            return

        # Initialize gradient if none is passed (defaulting to 1.0 for scalar output)
        if grad is None:
            if self.shape == (1,) or self.shape == ():
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError("Grad can only be implicitly created for scalar outputs")
        
        # Accumulate or initialize local gradient
        if self.grad is None:
            self.grad = np.asarray(grad, dtype=np.float32)
        else:
            self.grad += np.asarray(grad, dtype=np.float32)

        # Build topological sort of the graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v._creator is not None:
                    for input_node in v._creator.inputs:
                        build_topo(input_node)
                    topo.append(v)

        build_topo(self)

        # Traverse backward and invoke backward hooks
        for node in reversed(topo):
            if node._creator is None:
                continue
            
            # Call backward hook on the operation that created the node
            grads = node._creator.backward(node.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
                
            for parent, parent_grad in zip(node._creator.inputs, grads):
                if parent.requires_grad and parent_grad is not None:
                    # Resolve shape mismatch due to broadcasting
                    if parent_grad.shape != parent.shape:
                        parent_grad = self._unbroadcast(parent_grad, parent.shape)
                    
                    if parent.grad is None:
                        parent.grad = np.array(parent_grad, dtype=np.float32)
                    else:
                        parent.grad += np.array(parent_grad, dtype=np.float32)

    def _unbroadcast(self, grad, target_shape):
        """
        Helper method to sum out broadcasted dimensions to match target parent shape.
        """
        # Sum along prepended dimensions if grad ndim is larger
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
            
        # Sum along dimensions that were broadcasted (dimension size 1 in target)
        for axis, size in enumerate(target_shape):
            if size == 1:
                grad = grad.sum(axis=axis, keepdims=True)
                
        return grad

    # ------------------- Core Operator Mappings -------------------
    def __add__(self, other):
        from .ops import Add
        return Add.apply(self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        from .ops import Add, Neg
        return Add.apply(self, Neg.apply(other))

    def __rsub__(self, other):
        from .ops import Add, Neg
        return Add.apply(other, Neg.apply(self))

    def __mul__(self, other):
        from .ops import Mul
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        from .ops import MatMul
        return MatMul.apply(self, other)
        
    def relu(self):
        from .ops import ReLU
        return ReLU.apply(self)

    def sum(self, axis=None, keepdims=False):
        from .ops import Sum
        return Sum.apply(self, axis, keepdims)
