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
learning_rate = [0.01, 0.02, 0.05, 0.1, 0.5, 1.0]

# ---------------------------------------- #
# ---------------- CASE 1 ---------------- # 
# - Parametrize the learning rate
# - Use the ReLU activation function
# - Use the CrossEntropyLoss
# - Use the SGD optimizer
# - No regularization
# - 1 hidden layer with 30 neurons
# - Accuracy evaluation with the test data
# - 400 epochs
# ---------------------------------------- #

# Define the model
sizes = [784, 30, 10]
activations = [F.relu]
model = MLP(sizes, activations)

# Train the model for each learning rate
solution = []
for lr in learning_rate:
    solution.append(model.train_simple(inputs=db.train_inputs, labels=db.train_labels, epochs=epochs, 
                                    mini_batch_size=mini_batch_size, lr=lr, evaluation_data=evaluation_data))
    model.initialize_parameters()

# Add epochs vector to each solution
for i in range(len(solution)):
    solution[i]["epochs"] = np.arange(1, epochs+1)

# Save the solution to a text file in ./data folder
resultsFile = './data/noreg_varlr_hl30n_test.txt'
with open(resultsFile, 'w') as f:
    for s in solution:
        for key, value in s.items():
            f.write(key + '\n')
            f.write(';'.join(map(str, value)) + '\n')
        f.write('\n')


# ---------------------------------------- #
# ---------------- CASE 2 ---------------- #  
# - Parametrize the learning rate
# - Use the ReLU activation function
# - Use the CrossEntropyLoss
# - Use the SGD optimizer
# - No regularization
# - 1 hidden layer with 100 neurons
# - Accuracy evaluation with the test data
# - 400 epochs
# ---------------------------------------- #

# Define the model
sizes = [784, 100, 10]
activations = [F.relu]
model = MLP(sizes, activations)

# Train the model for each learning rate
solution = []
for lr in learning_rate:
    solution.append(model.train_simple(inputs=db.train_inputs, labels=db.train_labels, epochs=epochs, 
                                    mini_batch_size=mini_batch_size, lr=lr, evaluation_data=evaluation_data))
    model.initialize_parameters()

# Add epochs vector to each solution
for i in range(len(solution)):
    solution[i]["epochs"] = np.arange(1, epochs+1)

# Save the solution to a text file in ./data folder
resultsFile = './data/noreg_varlr_hl100n_test.txt'
with open(resultsFile, 'w') as f:
    for s in solution:
        for key, value in s.items():
            f.write(key + '\n')
            f.write(';'.join(map(str, value)) + '\n')
        f.write('\n')