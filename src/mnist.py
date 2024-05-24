import gzip, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

class Mnist(object):
    
    @staticmethod
    def vectorized_target(j, transpose=False):
        target = np.zeros((10, 1))
        target[j] = 1.0
        return target.T if transpose else target
    
    @staticmethod
    def random_shuffle(X, y):
        num_cols = X.shape[1]
        idx_cols = np.arange(num_cols)
        np.random.shuffle(idx_cols)
        return X[:, idx_cols], y[:, idx_cols]
    
    @staticmethod
    def plot_digit(image_data):
        image = np.reshape(image_data, (28, 28))
        plt.imshow(image, cmap="binary")
        plt.axis("off")
    
    def __init__(self, fileName, tensor=False):
        with gzip.open(fileName, 'rb') as f:
            training, validation, test = pickle.load(f, encoding='latin1')
        # Get the number of samples
        self.num_train_samples = training[0].shape[0]
        self.num_val_samples = validation[0].shape[0]
        self.num_test_samples = test[0].shape[0]
        # Get the inputs and outputs
        self.train_inputs = np.transpose(training[0])
        self.val_inputs = np.transpose(validation[0])
        self.test_inputs = np.transpose(test[0])
        self.train_outputs = np.hstack([Mnist.vectorized_target(y) for y in training[1]])
        self.val_outputs = np.hstack([Mnist.vectorized_target(y) for y in validation[1]])
        self.test_outputs = np.hstack([Mnist.vectorized_target(y) for y in test[1]])
        # Get the labels
        self.train_labels = training[1]
        self.val_labels = validation[1]
        self.test_labels = test[1]
        # Convert to tensor
        if tensor:
            self.train_inputs = torch.tensor(np.transpose(self.train_inputs), dtype=torch.float32)
            self.val_inputs = torch.tensor(np.transpose(self.val_inputs), dtype=torch.float32)
            self.test_inputs = torch.tensor(np.transpose(self.test_inputs), dtype=torch.float32)
            self.train_outputs = torch.tensor(np.transpose(self.train_outputs), dtype=torch.float32)
            self.val_outputs = torch.tensor(np.transpose(self.val_outputs), dtype=torch.float32)
            self.test_outputs = torch.tensor(np.transpose(self.test_outputs), dtype=torch.float32)
            self.train_labels = torch.tensor(self.train_labels, dtype=torch.long)
            self.val_labels = torch.tensor(self.val_labels, dtype=torch.long)
            self.test_labels = torch.tensor(self.test_labels, dtype=torch.long)

    def stats(self):
        text = []
        labels = [i for i in range(10)]
        set_labels = np.hstack([self.train_labels, self.val_labels, self.test_labels])
        num_by_label = [np.sum(set_labels==label) for label in labels]
        text.append("--------------------------------------------------\n")
        text.append("MNIST: Data base of handwritten digits from 0 to 9\n") # add headline
        text.append("--------------------------------------------------\n")
        text.append("Samples: Train({}) | Validation({}) | Test({})\n".
                    format(self.n_samples_train, self.n_samples_val, self.n_samples_test))
        text.append("Samples by number [0, 1, ..., 9]\n")
        for label, n_label in zip(labels, num_by_label):
            text.append(f"* Samples for {label} = {n_label}\n")
        text = ''.join(text)
        print(text)

    def filter(self, labels=[0]):
        # get indexes for each label
        train_filter_idx = [self.train_labels == label for label in labels]
        val_filter_idx = [self.val_labels == label for label in labels]
        test_filter_idx = [self.test_labels == label for label in labels]
        # extract inputs
        X_train_filter = [self.X_train[:,idxs] for idxs in train_filter_idx]
        X_val_filter = [self.X_val[:,idxs] for idxs in val_filter_idx]
        X_test_filter = [self.X_test[:,idxs] for idxs in test_filter_idx]
        # extract outputs
        y_train_filter = [self.y_train[:,idxs] for idxs in train_filter_idx]
        y_val_filter = [self.y_val[:,idxs] for idxs in val_filter_idx]
        y_test_filter = [self.y_test[:,idxs] for idxs in test_filter_idx]
        # stack data
        X_train = np.hstack(X_train_filter);    y_train = np.hstack(y_train_filter)
        X_val = np.hstack(X_val_filter);        y_val = np.hstack(y_val_filter)
        X_test = np.hstack(X_test_filter);      y_test = np.hstack(y_test_filter)
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)