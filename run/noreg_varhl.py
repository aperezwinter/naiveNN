import torch.nn.functional as F
import numpy as np
from src.nntorch import MLP
from src.mnist import Mnist

# Load the MNIST data
# File path is relative to the root folder
fileName = './data/mnist.pkl.gz'
db = Mnist(fileName, tensor=True)
evaluation_data=(db.test_inputs, db.test_labels)

# Define common hyperparameters
epochs = 400
mini_batch_size = 100
learning_rate = 0.01
hidden_layers = np.linspace(30, 100, 10, dtype=int, endpoint=True)
activations = [F.relu]

# Define the model and train it for each hidden layer size
solution = []
for hl in hidden_layers:
    sizes = [784, hl, 10]
    model = MLP(sizes, activations)
    solution.append(model.train_simple(inputs=db.train_inputs, labels=db.train_labels, epochs=epochs, 
                                       mini_batch_size=mini_batch_size, lr=learning_rate, evaluation_data=evaluation_data))
    model.initialize_parameters()

# Add epochs vector to each solution
for i in range(len(solution)):
    solution[i]["epochs"] = np.arange(1, epochs+1)

# Save the solution to a text file in ./data folder
resultsFile = './data/noreg_varhl_test.txt'
with open(resultsFile, 'w') as f:
    for s in solution:
        for key, value in s.items():
            f.write(key + '\n')
            f.write(';'.join(map(str, value)) + '\n')
        f.write('\n')