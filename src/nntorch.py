import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class MLP(nn.Module):
    def __init__(self, sizes, activations, loss_fn=nn.CrossEntropyLoss, params='uniform'):
        super(MLP, self).__init__()
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.activations = activations
        self.loss_fn = loss_fn()
        # Use nn.ModuleList to register the layers
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.initializeParameters(params)
        # Set device: GPU / CPU (Mac version)
        self.device = torch.device("cpu")
        #self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def __call__(self, x):
        for i, activation in enumerate(self.activations):
            x = activation(self.layers[i](x))
        if self.loss_fn != nn.CrossEntropyLoss:
            return F.softmax(self.layers[-1](x), dim=1)
        else:
            return self.layers[-1](x)

    def initializeParameters(self, params='uniform'):
        if params == 'uniform':
            for layer in self.layers:
                nn.init.uniform_(layer.weight, a=-1.0, b=1.0)
                nn.init.constant_(layer.bias, 0.0)
        else:
            for layer in self.layers:
                nn.init.normal_(layer.weight, mean=0.0, std=1.0)
                nn.init.constant_(layer.bias, 0.0)

    def getParametersByLayer(self):
        return [layer.parameters() for layer in self.layers]

    def getParametersNorm(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        norm = sum([torch.norm(param).item() for param in self.parameters()])
        return norm / num_params

    def getParametersNormByLayer(self):
        num_params_ly = [sum(p.numel() for p in layer.parameters() if p.requires_grad) for layer in self.layers]
        norm_ly = [sum([torch.norm(param).item() for param in layer.parameters()]) for layer in self.layers]
        return [norm / num_params for norm, num_params in zip(norm_ly, num_params_ly)]

    def _trainBatch(self, inputs, labels, optimizer):
        outputs = self(inputs)                    # forward pass
        loss = self.loss_fn(outputs, labels)      # compute the training loss
        optimizer.zero_grad()                     # zero the gradients
        loss.backward()                           # backward pass
        optimizer.step()                          # update the parameters (weights and biases)
        return loss.item()
    
    def _evaluate(self, train_batches:list, eval_batches:list, save:dict, monitor_train:bool = True, 
                  monitor_params:bool = False, monitor_params_layer:bool = False):
        if monitor_train:
            save['loss'][0].append(self.loss(data_batches=train_batches))
            save['accuracy'][0].append(self.accuracy(data_batches=train_batches))
        if eval_batches:
            save['loss'][1].append(self.loss(data_batches=eval_batches))
            save['accuracy'][1].append(self.accuracy(data_batches=eval_batches))
        if monitor_params:
            save['norm'][0].append(self.getParametersNorm())
        if monitor_params_layer:
            save['norm'][1].append(self.getParametersNormByLayer())
        return save

    def predict(self, inputs):
        outputs = self(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted
    
    def train(self, train_batches, optimizer=optim.SGD, epochs=30, lr=0.01, regularization=0.0,
              eval_batches=None, monitor_train=False, monitor_params=False, monitor_params_layer=False, 
              only_last=False, verbose=False, epch_print=1):
        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=regularization)
        if only_last:
            values = {'loss': ([],[]), 'accuracy': ([],[]), 'norm': ([],[])}
            for i in range(epochs):
                for (inputs, labels) in train_batches:
                    if self.device.type != 'cpu':
                        inputs = inputs.to(self.device) # push inputs to GPU
                        labels = labels.to(self.device) # push labels to GPU
                    self._trainBatch(inputs, labels, optimizer)
                if i == epochs-1:
                    values = self._evaluate(train_batches, eval_batches, values, 
                                            monitor_train, monitor_params, monitor_params_layer)
        else:
            values = {'loss': ([],[]), 'accuracy': ([],[]), 'norm': ([],[])}
            for i in range(epochs):
                for (inputs, labels) in train_batches:
                    if self.device.type != 'cpu':
                        inputs = inputs.to(self.device)  # push inputs to GPU
                        labels = labels.to(self.device)  # push labels to GPU
                    self._trainBatch(inputs, labels, optimizer)
                values = self._evaluate(train_batches, eval_batches, values, 
                                        monitor_train, monitor_params, monitor_params_layer)
                if verbose and (i+1) % epch_print == 0:
                    print("Epoch {:2d}/{}: Loss ({:.4g}, {:.4g}) \t Accuracy ({:.2f}%, {:.2f}%)".format(
                        i+1, epochs, values['loss'][0][-1], values['loss'][1][-1],
                        100*values['accuracy'][0][-1], 100*values['accuracy'][1][-1]))
        return values

    def loss(self, data_batches):
        X = torch.vstack([inputs_batch for inputs_batch, _ in data_batches])
        y_gt = torch.hstack([labels_batch for _, labels_batch in data_batches])
        with torch.no_grad():
            if self.device.type != 'cpu':
                X = X.to(self.device)       # push inputs to GPU
                y_gt = y_gt.to(self.device) # push labels to GPU
            y = self(X)
        return self.loss_fn(y, y_gt).item()
    
    def accuracy(self, data_batches):
        X = torch.vstack([inputs_batch for inputs_batch, _ in data_batches])
        y_gt = torch.hstack([labels_batch for _, labels_batch in data_batches])
        with torch.no_grad():
            if self.device.type != 'cpu':
                X = X.to(self.device)       # push inputs to GPU
                y_gt = y_gt.to(self.device) # push labels to GPU
            y_pred = self.predict(X)
        accuracy = (y_pred == y_gt).sum().item() / y_gt.size(0)
        return accuracy
    
    def precision(self, data_batches):
        X = torch.vstack([inputs_batch for inputs_batch, _ in data_batches])
        y_gt = torch.hstack([labels_batch for _, labels_batch in data_batches])
        with torch.no_grad():
            if self.device.type != 'cpu':
                X = X.to(self.device)       # push inputs to GPU
                y_gt = y_gt.to(self.device) # push labels to GPU
            y_pred = self.predict(X)
        return precision_score(y_gt, y_pred, average='macro')
    
    def recall(self, data_batches):
        X = torch.vstack([inputs_batch for inputs_batch, _ in data_batches])
        y_gt = torch.hstack([labels_batch for _, labels_batch in data_batches])
        with torch.no_grad():
            if self.device.type != 'cpu':
                X = X.to(self.device)       # push inputs to GPU
                y_gt = y_gt.to(self.device) # push labels to GPU
            y_pred = self.predict(X)
        return recall_score(y_gt, y_pred, average='macro')
    
    def f1(self, data_batches):
        X = torch.vstack([inputs_batch for inputs_batch, _ in data_batches])
        y_gt = torch.hstack([labels_batch for _, labels_batch in data_batches])
        with torch.no_grad():
            if self.device.type != 'cpu':
                X = X.to(self.device)       # push inputs to GPU
                y_gt = y_gt.to(self.device) # push labels to GPU
            y_pred = self.predict(X)
        return f1_score(y_gt, y_pred, average='macro')