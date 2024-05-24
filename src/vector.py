import numpy as np
from src.scalar import Scalar

class Vector:
    """Tensor of rank 1: stores their values and gradients. """

    def __init__(self, value, _children=(), _op='', _label=''):
        self.size = len(value)
        self.value = np.array(value, dtype=float)
        self.grad = np.zeros((self.size, self.size), dtype=float)
        # private variables for build autograd graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # operation produced by this node, for graphviz, debugging, etc
        self._label = _label # scalar label, for graphviz, debugging, etc

    def __add__(self, other): # self + other
        assert (self.size == other.size), f"size-mismatch for sum, {self.size}!={other.size}"
        other = other if isinstance(other, Vector) else Vector(other)
        out = Vector(value=self.value + other.value, _children=(self, other), _op='+')

        def _backward(): # local backward
            # Let z = x + y, then dz_i/dx_j = dz_i/dy_j = delta_ij
            # So, dz/dx = dz/dz * dz/dx = I * I = out.grad * I
            # So, dz/dy = dz/dz * dz/dy = I * I = out.grad * I
            self.grad += np.identity(out.size)
            other.grad += np.identity(out.size)
        out._backward = _backward # set backward func from '+' operation

        return out
    
    def __sub__(self, other): # self - other
        assert (self.size == other.size), f"size-mismatch for substract, {self.size}!={other.size}"
        other = other if isinstance(other, Vector) else Vector(other)
        out = Vector(value=self.value - other.value, _children=(self, other), _op='-')

        def _backward(): # local backward
            # Let z = x - y, then dz_i/dx_j = delta_ij, dz_i/dy_j = -delta_ij
            # So, dz/dx = dz/dz * dz/dx = I * I = out.grad * I
            # So, dz/dy = dz/dz * dz/dy = I * -I = out.grad * -I
            self.grad += np.identity(out.size)
            other.grad -= np.identity(out.size)
        out._backward = _backward # set backward func from '-' operation

        return out
    
    def __mul__(self, other): # self * other
        other = Vector(value=[other]*self.size) if isinstance(other, (int, float)) else other
        assert (self.size == other.size), f"size-mismatch for product, {self.size} != {other.size}"
        out = Vector(value=self.value * other.value, _children=(self, other), _op='*')

        def _backward(): # local backward
            # Let z = x*y, so zi = xi*yi. Then dzi/dxj = yj*delta_ij = {yi,0}
            # and dzi/dyj = xj*delta_ij = {xi,0}. So, gradx_z = diag(y) and grady_z = diag(x).
            self.grad += out.grad * np.diag(other.value)
            other.grad += out.grad * np.diag(self.value)
        out._backward = _backward

        return out

    def __truediv__(self, other): # self / other
        # Check zero division problem
        if isinstance(other, (int, float)):
            assert other != 0, "Division by zero!"
            other = Vector(other * np.ones(self.size))
        elif isinstance(other, Vector):
            assert (np.any(other.value == 0)), "Division by zero!"
        else:
            raise TypeError(f"type-mismatch, must be (int, float, Vector) not {type(other)}.")
        out = Vector(value=self.value / other.value, _children=(self, other), _op='/')

        def _backward(): # local backward
            # Let z = x / y, so zi = xi/yi.
            # Then, dzi/dxj = 1/yj * delta_ij = {1/yi,0} -> gradx_z = diag(1/y).
            # Then, dzi/dyj = -xj/yj^2 * delta_ij = {-xi/yi^2,0} -> grady_z = diag(-x/y^2)
            self.grad += out.grad * np.diag(other.value**-1)
            other.grad -= out.grad * np.diag(self.value/other.value**2)
        out._backward = _backward

        return out
    
    def __neg__(self): # -self
        return self * -1
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __rsub__(self, other): # other - self
        return other - self
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __rtruediv__(self, other): # other / self
        if isinstance(other, (int, float)):
            other = Vector(other * np.ones(self.size))
        elif isinstance(other, Vector):
            pass
        else:
            raise TypeError(f"type-mismatch, must be (int, float, Vector) not {type(other)}.")
        
        return other / self
    
    def __str__(self):
        out = f"Vector({self._label}):\n"
        out += f"* value = {self.value}\n"
        out += f"* grad  = {self.grad}"
        return out
    
    def __repr__(self):
        return f"Vector('{self._label}', size={self.size})"
    
    def backward(self):
        # Topological order all of the children in the graph
        topo, visited = [], set()
        def build_topo(vector):
            if vector not in visited:
                visited.add(vector)
                for child in vector._prev:
                    build_topo(child)
                topo.append(vector)
        build_topo(self)

        # Set grad=1 and go one variable at a time and apply 
        # the chain rule to get its gradient
        self.grad = np.identity(self.size)
        for vector in reversed(topo):
            vector._backward()

    def dot(self, other): # self . other
        # Dot product between two vectors
        assert (self.size == other.size), f"size-mismatch for dot product, {self.size} != {other.size}"
        out = Scalar(value=np.dot(self.value, other.value), _children=(self, other), _op='.')

        def _backward(): # local backward
            # Let a = x.y = xi*yi, with x,y in R^n
            # Then, da/dxj = yi * delta_ij = yj, so da/dx = y
            # Then, da/dyj = xi * delta_ij = xj, so da/dy = x
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value
        out._backward = _backward

        return out 