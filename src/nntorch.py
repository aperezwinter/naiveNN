import torch
import torch.nn as nn
import torch.optim as optim

    
class MLP(nn.Module):
    def __init__(self, sizes, activations, loss_fn=nn.CrossEntropyLoss):
        super(MLP, self).__init__()
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.layers = [nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
        self.activations = activations
        self.loss_fn = loss_fn()
        # Initialize the weights and biases
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.constant_(layer.bias, 0.0)
    
    def __call__(self, x):
        for i, activation in enumerate(self.activations):
            x = activation(self.layers[i](x))
        return self.layers[-1](x)
    
    def initialize_parameters(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.constant_(layer.bias, 0.0)
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
    def parameters_by_layer(self):
        return [layer.parameters() for layer in self.layers]
    
    def parameters_norm(self):
        return sum([torch.norm(param).item() for param in self.parameters()])
    
    def parameters_norm_by_layer(self):
        return [sum([torch.norm(param).item() for param in params]) for params in self.parameters_by_layer()]
    
    def train_simple(self, inputs, labels, optimizer=optim.SGD, epochs=30, mini_batch_size=10, 
                   lr=0.01, regularization=0.0, evaluation_data=None):
        solution = {'evaluation_cost': [], 'evaluation_accuracy': [], 'training_cost': [], 'training_accuracy': []}
        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=regularization)
        for _ in range(epochs):
            # Run the mini-batch stochastic gradient descent
            running_loss = 0.0
            for i in range(0, len(inputs), mini_batch_size):    # for each mini-batch
                x = inputs[i:i+mini_batch_size,:]               # get the inputs
                y = labels[i:i+mini_batch_size]                 # get the labels
                optimizer.zero_grad()                           # zero the gradients
                outputs = self(x)                               # forward pass
                loss = self.loss_fn(outputs, y)                 # compute the loss
                loss.backward()                                 # backward pass
                optimizer.step()                                # update the parameters (weights and biases)
                running_loss += loss.item()                     # sum the loss
            # Compute and save the cost and accuracy
            solution['training_cost'].append(running_loss/len(inputs))
            solution['training_accuracy'].append(self.evaluate(inputs, labels))
            if evaluation_data:
                solution['evaluation_cost'].append(self.loss_fn(self(evaluation_data[0]), evaluation_data[1]).item())
                solution['evaluation_accuracy'].append(self.evaluate(evaluation_data[0], evaluation_data[1]))
        return solution
    
    def train(self, inputs, labels, optimizer=optim.SGD, epochs=30, mini_batch_size=10, lr=0.01, regularization=0.0, 
              evaluation_data=None, monitor_evaluation_cost=False, monitor_evaluation_accuracy=False, 
              monitor_training_cost=False, monitor_training_accuracy=False, monitor_parameters=False, 
              monitor_parameters_by_layer=False, verbose=False, epoch_print=1):
        solution = {'evaluation_cost': [], 'evaluation_accuracy': [], 'training_cost': [], 
                    'training_accuracy': [], 'parameters_norm': [], 'parameters_norm_by_layer': []}
        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=regularization)
        for epoch in range(epochs):
            # Run the mini-batch stochastic gradient descent
            running_loss = 0.0
            for i in range(0, len(inputs), mini_batch_size):    # for each mini-batch
                x = inputs[i:i+mini_batch_size,:]               # get the inputs
                y = labels[i:i+mini_batch_size]                 # get the labels
                optimizer.zero_grad()                           # zero the gradients
                outputs = self(x)                               # forward pass
                loss = self.loss_fn(outputs, y)                 # compute the loss
                loss.backward()                                 # backward pass
                optimizer.step()                                # update the parameters (weights and biases)
                running_loss += loss.item()                     # sum the loss
            # Compute and save the cost and accuracy
            if monitor_training_cost:
                solution['training_cost'].append(running_loss/len(inputs))
            if monitor_training_accuracy:
                solution['training_accuracy'].append(self.evaluate(inputs, labels))
            if evaluation_data:
                if monitor_evaluation_cost:
                    solution['evaluation_cost'].append(self.loss_fn(self(evaluation_data[0]), evaluation_data[1]).item())
                if monitor_evaluation_accuracy:
                    solution['evaluation_accuracy'].append(self.evaluate(evaluation_data[0], evaluation_data[1]))
            # Compute and save the parameters norm
            if monitor_parameters_by_layer:
                solution['parameters_norm_by_layer'].append(self.parameters_norm_by_layer())
                if monitor_parameters:
                    solution['parameters_norm'].append(sum(solution['parameters_norm_by_layer'][-1]))
            elif monitor_parameters:
                solution['parameters_norm'].append(self.parameters_norm())
            else: pass
            # Print the cost and accuracy
            if verbose and ((epoch+1) % epoch_print == 0):
                print("Epoch {}/{} - Loss: {:.4g} - Accuracy: {:.4g}%".format(
                    epoch+1, epochs, running_loss/len(inputs), solution['training_accuracy'][-1]))
        return solution
    
    def evaluate(self, inputs, labels):
        outputs = self(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy