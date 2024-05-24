import numpy as np
from .mnist import Mnist

class Linear(object):
    @staticmethod
    def eval(z):
        return z
    
    @staticmethod
    def grad(z):
        return np.ones(z.shape)
    
class Relu(object):
    @staticmethod
    def eval(z):
        return np.maximum(z, 0)
    
    @staticmethod
    def grad(z):
        return np.ones(z.shape) * (z >= 0)

class LeakyRelu(object):
    @staticmethod
    def eval(z, alpha=0.01):
        return np.maximum(z, alpha * z)
    
    @staticmethod
    def grad(z, alpha=0.01):
        return np.ones(z.shape) * (z >= 0) + alpha * (z < 0)

class Sigmoid(object):
    @staticmethod
    def eval(z):
        ez = np.exp(z)
        return ez / (1.0 + ez)
    
    @staticmethod
    def grad(z):
        sigma = Sigmoid.eval(z)
        return sigma * (1 - sigma)
    
class Tanh(object):
    @staticmethod
    def eval(z):
        return np.tanh(z)
    
    @staticmethod
    def grad(z):
        return 1 - np.tanh(z)**2
    
class Softmax(object):
    @staticmethod
    def eval(z):
        ez = np.exp(z)
        sum_ez = np.reshape(np.sum(ez, axis=0), (1, ez.shape[1]))
        sum_ez = np.repeat(sum_ez, ez.shape[0], axis=0)
        return ez / sum_ez
    
    @staticmethod
    def grad(z):
        softmax = Softmax.eval(z)
        return softmax * (1 - softmax)
    
class MSECost(object):
    @staticmethod
    def eval(act, y):
        return 0.5 * np.linalg.norm(act - y)**2 / y.shape[1]
    
    @staticmethod
    def grad(act, y):
        return act - y
    
class CrossEntropyCost(object):
    @staticmethod
    def eval(act, y):
        log_act = np.nan_to_num(np.log(act))
        log_1_act = np.nan_to_num(np.log(1-act))
        return np.sum(-y*log_act-(1-y)*log_1_act) / y.shape[1]
    
    @staticmethod
    def grad(act, y):
        return np.nan_to_num((act - y) / (act * (1 - act)))

class Layer(object):
    def __init__(self, nin, nout, activation=Relu, init_params="unit"):
        self.nin = nin
        self.nout = nout
        self.act = activation
        self.grad_bias = np.zeros((nout, 1))
        self.grad_weigths = np.zeros((nout, nin))
        self.params_initializer(option=init_params)
        
    def params_initializer(self, option="unit"):
        if option == "default":
            self.bias = np.random.randn(self.nout, 1)
            self.weights = np.random.randn(self.nout, self.nin)/np.sqrt(self.nin)
        elif option == "unit":
            self.bias = np.random.uniform(-1, 1, size=(self.nout, 1))
            self.weights = np.random.uniform(-1, 1, size=(self.nout, self.nin))
        else:
            self.bias = np.random.randn(self.nout, 1)
            self.weights = np.random.randn(self.nout, self.nin)

    def __call__(self, X):
        bias = np.repeat(self.bias, X.shape[1], axis=1)
        return self.act.eval(self.weights @ X + bias)
    
    def __repr__(self):
        return f"Layer of [{self.nin},{self.nout}]"
    
    def params(self):
        return [self.weights, self.bias]
    
    def params_norm(self):
        return np.linalg.norm(self.weights) + np.linalg.norm(self.bias)
    
    def z_out(self, X):
        bias = np.repeat(self.bias, X.shape[1], axis=1)
        return self.weights @ X + bias
    
    def zero_grad(self):
        self.grad_bias = np.zeros((self.nout, 1))
        self.grad_weigths = np.zeros((self.nout, self.nin))

class MLP(object):
    def __init__(self, layers, cost=MSECost):
        self.num_layers = len(layers)
        self.layers = layers
        self.cost = cost

    def feedfoward(self, X):
        zs = []
        activation = X
        activations = [X]
        for layer in self.layers:
            z = layer.z_out(activation)
            zs.append(z)
            activation = layer.act.eval(z)
            activations.append(activation)
        return zs, activations
    
    def backprop(self, X, y):
        """X.shape = (N, M)
        N = number of inputs
        M = number of samples
        
        y.shape = (K, M)
        K = number of outputs
        M = number of samples
        """
        # feedfoward
        zs, activations = self.feedfoward(X)
        # backward pass
        last_layer = (self.layers[-1].act == Sigmoid) or (self.layers[-1].act == Softmax)
        delta = [activations[-1] - y] if last_layer else [self.cost.grad(activations[-1], y) * self.layers[-1].act.grad(zs[-1])]
        for l in range(2, self.num_layers+1):
            delta.insert(0, np.transpose(self.layers[-l+1].weights) @ delta[-l+1] * self.layers[-l].act.grad(zs[-l]))
        # set gradients to zero
        for layer in self.layers:
            layer.zero_grad()
        # coompute gradients
        for i, delta_i in enumerate(delta):
            self.layers[i].grad_bias += np.reshape(delta_i[:,0], (delta_i.shape[0], 1))
            self.layers[i].grad_weigths += delta_i @ np.transpose(activations[i])
    
    def predict(self, X):
        output = X  # inputs are the first outputs
        for layer in self.layers:
            output = layer(output)
        return output
    
    def update_mini_batch(self, X, y, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        m = X.shape[1] # mini batch length
        self.backprop(X, y)
        for layer in self.layers:
            layer.bias -= eta * layer.grad_bias / m
            layer.weights -= eta * (lmbda * layer.weights / n + layer.grad_weigths / m)

    def fit(self, X, y, epochs, mini_batch_size, eta, lmbda = 0.0, evaluation_data=None, verbose=False):
        n = X.shape[1]
        n_eval = evaluation_data[0].shape[1] if evaluation_data else 1
        train_cost = np.zeros(epochs)
        train_acc = np.zeros(epochs)
        eval_cost = np.zeros(epochs)
        eval_acc = np.zeros(epochs)
        for j in range(epochs):
            X_shuffle, y_shuffle = Mnist.random_shuffle(X, y)
            mini_batches = [(X_shuffle[:,k:k+mini_batch_size], y_shuffle[:,k:k+mini_batch_size]) \
                            for k in range(0, n, mini_batch_size)]
            for X_batch, y_batch in mini_batches:
                self.update_mini_batch(X_batch, y_batch, eta, lmbda, n)
            train_cost[j] = self.total_cost(X_shuffle, y_shuffle, lmbda)
            train_acc[j] = self.accuracy(X_shuffle, y_shuffle)
            if evaluation_data:
                eval_cost[j] = self.total_cost(evaluation_data[0], evaluation_data[1], lmbda)
                eval_acc[j] = self.accuracy(evaluation_data[0], evaluation_data[1])
            if verbose:
                print("Epoch {}. Train: ({:.5g}, {:.5g}%) | Eval: ({:.5g}, {:.5g}%)".
                      format(j, train_cost[j], 100*train_acc[j]/n, eval_cost[j], 100*eval_acc[j]/n_eval))
        
        return (train_cost, eval_cost), (train_acc, eval_acc)
    
    def accuracy(self, X, y):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        """
        return np.sum(np.argmax(self.predict(X), axis=0) == np.argmax(y, axis=0))
    
    def total_cost(self, X, y, lmbda):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        m = X.shape[1]  # number of samples
        w_norm = 0  # weights norm
        out = self.predict(X)
        if lmbda != 0:
            w_norm = sum([np.linalg.norm(layer.weights)**2 for layer in self.layers])
        cost = self.cost.eval(out, y) + 0.5 * lmbda * w_norm / m
        return cost