import numpy as np
import matplotlib.pyplot as plt

class Tensor:

    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data, dtype=np.float32)  # Fix: grad starts at zero
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor({self.data.shape}, grad = {self.grad.shape}, op = {self._op})"

    def __matmul__(self, other):  # Fix: was _matmul_ (single underscores)
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad  # Fix: @ not *
        out._backward = _backward
        return out

    def _unbroadcast(self, grad, target_shape):
        """Collapses a broadcasted gradient back to its original tensor shape."""
        # 1. Sum away extra dimensions added to the front
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)
        # 2. Sum along dimensions that were broadcasted from 1 to N
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[i] > 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            grad_self = out.grad
            grad_other = out.grad

            self.grad += self._unbroadcast(grad_self, self.data.shape)
            other.grad += self._unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # Fix: multiplication rule — each grad uses the *other* operand's data
            self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** power, (self,), f'**{power}')  # Fix: was 'other', should be 'power'

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad  # Fix: same here
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**(-1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other  # Fix: was 'other - self' which would infinite loop

    # ── Shape property ──────────────────────────────────────────────────
    @property
    def shape(self):
        return self.data.shape

    # ── Elementwise math ops ────────────────────────────────────────────
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad  # d/dx e^x = e^x
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # ── Reduction ops ───────────────────────────────────────────────────
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            # Broadcast the gradient back to the original shape
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape).copy()
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), 'mean')

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad / n, self.data.shape).copy()
        out._backward = _backward
        return out

    # ── Activation functions ────────────────────────────────────────────
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0).astype(np.float32) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    # ── Gradient utilities ──────────────────────────────────────────────
    def zero_grad(self):
        """Reset gradients to zero for this tensor and all ancestors."""
        visited = set()
        def _reset(t):
            if t not in visited:
                visited.add(t)
                t.grad = np.zeros_like(t.data, dtype=np.float32)
                for child in t._prev:
                    _reset(child)
        _reset(self)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=np.float32)
        for node in reversed(topo):
            node._backward()
