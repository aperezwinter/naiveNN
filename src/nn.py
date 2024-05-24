import random
from graphviz import Digraph
from src.scalar import Scalar
from src.vector import Vector

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        if isinstance(n, Scalar):
            dot.node(name=str(id(n)), label = "{%s (%.3g, %.3g)}" % (n._label, n.value, n.grad), shape='record')
        elif isinstance(n, Vector):
            dot.node(name=str(id(n)), label = f"{n._label} ({n.value}, {n.grad})", shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        # nin: number of inputs
        super().__init__()
        self.weights = [Scalar(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Scalar(0)
        self.nonlin = nonlin

    def __call__(self, x):
        a = sum((wi*xi for wi,xi in zip(self.weights, x)), self.bias)
        return a.relu() if self.nonlin else a.linear()
        
    def parameters(self):
        return self.weights + [self.bias]
    
    def __repr__(self) -> str:
        return f"{self.act} Neuron({len(self.weights)})"
    
class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        # nin: number of inputs
        # nout: number of outputs
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):

    def __init__(self, nin, nouts):
        # nin: number of inputs (input layer)
        # nout: number of outputs (hidden + out layers)
        # nonlin=True for all hidden layers, except for the last layer
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        self._loss = Scalar(1.0)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
    def backward(self):
        self.zero_grad() # flush gradients (set to zero)
        self._loss.backward()

    def predict(self, X):
        ypred = [self(x) for x in X]
        return ypred
    
    def loss(self, X, y, out=True):
        ypred = self.predict(X)
        self._loss = sum((yout - ygt)**2 for ygt, yout in zip(y, ypred))
        return self._loss if out else None

    def fit(self, X, y, alpha=1e-2, maxiter=10000, tol=1e-6):
        i=0
        while (i < maxiter) and (self._loss.value > tol):
            self.loss(X, y) # forward pass
            self.backward() # backward pass
            for p in self.parameters(): # update
                p.value -= alpha * p.grad # gradient descent
            i += 1  