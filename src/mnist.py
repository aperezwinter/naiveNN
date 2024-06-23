import gzip, pickle, torch
import numpy as np
import pandas as pd
import seaborn as sns
import albumentations as A
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class Mnist(object):
    
    @staticmethod
    def oneHotEncoding(array, num_classes=10, transpose=False):
        # check if array is a tensor
        if isinstance(array, torch.Tensor):
            tensor = torch.nn.functional.one_hot(array, num_classes)
        else:
            tensor = torch.nn.functional.one_hot(torch.tensor(array), num_classes)
        return tensor.T if transpose else tensor
    
    @staticmethod
    def randomShuffle(inputs, labels):
        n_samples = inputs.shape[0]
        idx_samples = np.arange(n_samples)
        np.random.shuffle(idx_samples)
        return inputs[idx_samples], labels[idx_samples]
    
    @staticmethod
    def plotDigit(input):
        input = np.reshape(input, (28, 28))
        plt.imshow(input, cmap="gray")
        plt.axis("off")
        plt.grid(False)

    @staticmethod
    def plotDigits(inputs, max_cols=5, file_name=None, title=None, figsize=(8, 8)):
        n_digits = inputs.shape[0]
        n_cols = min(n_digits, max_cols)
        n_rows = n_digits // n_cols
        plt.figure(figsize=figsize)
        for i in range(n_digits):
            plt.subplot(n_rows, n_cols, i+1)
            Mnist.plotDigit(inputs[i])
        plt.tight_layout()
        if title:
            plt.suptitle(title)
        if file_name:
            plt.savefig(file_name)

    @staticmethod
    def transform(shift=0.0625, scale=0.1, rotate=20.0, p=1.0):
        return A.ShiftScaleRotate(shift_limit=shift, scale_limit=scale, 
                                  rotate_limit=rotate, p=p)

    def __init__(self, fileName, tensor=False):
        # Load data
        with gzip.open(fileName, 'rb') as f:
            train, validation, test = pickle.load(f, encoding='latin1')
        # Define scalar values
        self.n_samples_train = train[0].shape[0]
        self.n_samples_val = validation[0].shape[0]
        self.n_samples_test = test[0].shape[0]
        self.n_samples = train[0].shape[0] + validation[0].shape[0] + test[0].shape[0]
        self.n_features = train[0].shape[1]
        self.n_classes = len(set(train[1]))
        self.classes = list(set(train[1]))
        # Define datasets (inputs, labels)
        self.X_train = train[0] 
        self.y_train = train[1]
        self.X_val = validation[0]
        self.y_val = validation[1]
        self.X_test = test[0]
        self.y_test = test[1]

    def getBatches(self, inputs, labels, batch_size:int = 10, tensor:bool = False):
        n_batches = inputs.shape[0] // batch_size
        inputs_batches = np.array_split(inputs, n_batches, axis=0)
        labels_batches = np.array_split(labels, n_batches)
        if tensor:
            batches = [(torch.tensor(X), torch.tensor(y)) for X, y in zip(inputs_batches, labels_batches)]
        else:
            batches = [(X, y) for X, y in zip(inputs_batches, labels_batches)]
        return batches
    
    def getMoments(self, split=False):
        if split:
            moments = []
            moments.append((np.mean(self.X_train.reshape(-1)), 
                            np.mean(self.X_val.reshape(-1)), 
                            np.mean(self.X_test.reshape(-1))))
            moments.append((np.std(self.X_train.reshape(-1)), 
                            np.std(self.X_val.reshape(-1)), 
                            np.std(self.X_test.reshape(-1))))
            moments.append((skew(self.X_train.reshape(-1)), 
                            skew(self.X_val.reshape(-1)), 
                            skew(self.X_test.reshape(-1))))
            moments.append((kurtosis(self.X_train.reshape(-1), fisher=False), 
                            kurtosis(self.X_val.reshape(-1), fisher=False), 
                            kurtosis(self.X_test.reshape(-1), fisher=False)))
        else:
            X = np.vstack([self.X_train, self.X_val, self.X_test]).reshape(-1)
            moments = [np.mean(X), np.std(X), skew(X.ravel()), kurtosis(X.ravel(), fisher=False)]
        return moments
    
    def filter(self, label:int = 0, set:str = "train"):
        if set == "train":
            X_train_filtered = self.X_train[self.y_train == label]
            y_train_filtered = self.y_train[self.y_train == label]
            return X_train_filtered, y_train_filtered
        elif set == "val":
            X_val_filtered = self.X_val[self.y_val == label]
            y_val_filtered = self.y_val[self.y_val == label]
            return X_val_filtered, y_val_filtered
        elif set == "test":
            X_test_filtered = self.X_test[self.y_test == label]
            y_test_filtered = self.y_test[self.y_test == label]
            return X_test_filtered, y_test_filtered
        else:
            X = np.vstack([self.X_train, self.X_val, self.X_test])
            y = np.hstack([self.y_train, self.y_val, self.y_test])
            X_filtered = X[y == label]
            y_filtered = y[y == label]
            return X_filtered, y_filtered
        
    def augment(self, inputs, labels, n_aug:int = 10, shift:float = 0.0625, 
                scale:float = 0.1, rotate:float = 20.0, p:float = 1.0):
        inputs_aug = []
        labels_aug = []
        trasform = Mnist.transform(shift, scale, rotate, p)
        for Xi, yi in zip(inputs, labels):
            Xi = Xi.reshape(28, 28)
            X_aug = [trasform(image=Xi)['image'] for _ in range(n_aug)]
            X_aug = np.vstack([Xi.reshape(1, -1) for Xi in X_aug])
            y_aug = np.repeat(yi, n_aug)
            inputs_aug.append(X_aug)
            labels_aug.append(y_aug)
        inputs_aug = np.vstack(inputs_aug)
        labels_aug = np.hstack(labels_aug)
        return inputs_aug, labels_aug
        
    def barPlot(self, title:str = None, file_name:str = None, figsize=(6, 4)):
        y = np.hstack([self.y_train, self.y_val, self.y_test])
        labels, counts = np.unique(y, return_counts=True)

        title = title if title else "Digit distribution"
        plt.figure(figsize=figsize)
        sns.barplot(x=labels, y=counts, palette='viridis', hue=labels, legend=False)
        plt.xlabel('Digit', fontsize=12)
        plt.ylabel('Amount', fontsize=12)
        plt.title(title, fontsize=16)
        plt.xticks(labels, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(file_name) if file_name else plt.show()
    
    def boxPlot(self, title:str = None, file_name:str = None, figsize=(6, 4)):
        X = np.vstack([self.X_train, self.X_val, self.X_test])
        y = np.hstack([self.y_train, self.y_val, self.y_test])

        mean_digits = {}; distances = []
        for digit in self.classes:
            mean_digits[digit] = X[y == digit].mean(axis=0)
        for Xi, yi in zip(X, y):
            distance = np.linalg.norm(Xi - mean_digits[yi])
            distances.append({"digit": yi, "distance": distance})
        df = pd.DataFrame(distances)

        title = title if title else "Relative digit variance"
        plt.figure(figsize=figsize)
        sns.boxplot(x='digit', y='distance', data=df, palette='viridis', hue='digit', legend=False)
        plt.xlabel('Digit', fontsize=12)
        plt.ylabel('L2 norm', fontsize=12)
        plt.title(title, fontsize=16)
        plt.savefig(file_name) if file_name else plt.show()

    def plotCumulativePCA(self, n_components:int = 784, title:str = None, file_name:str = None):
        X = np.vstack([self.X_train, self.X_val, self.X_test])
        y = np.hstack([self.y_train, self.y_val, self.y_test])

        pca = PCA(n_components=n_components)
        pca.fit(X)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        n_comp_90 = np.argmax(explained_variance >= 0.90) + 1
        n_comp_95 = np.argmax(explained_variance >= 0.95) + 1
        n_comp_99 = np.argmax(explained_variance >= 0.99) + 1

        title = title if title else "Cumulative explained variance"
        plt.figure(figsize=(6, 4))
        plt.plot(explained_variance, color='black', lw=3)
        plt.axhline(0.90, c='red', linestyle='--', lw=1, label=f"90% - {n_comp_90} components")
        plt.axhline(0.95, c='green', linestyle='--', lw=1, label=f"95% - {n_comp_95} components")
        plt.axhline(0.99, c='blue', linestyle='--', lw=1, label=f"99% - {n_comp_99} components")
        plt.xlabel('Number of components', fontsize=12)
        plt.ylabel('Cumulative explained variance', fontsize=12)
        plt.title(title, fontsize=16)
        plt.savefig(file_name) if file_name else plt.show()

    def getNumComponentsPCA(self, variance:float = 0.99):
        X = np.vstack([self.X_train, self.X_val, self.X_test])
        pca = PCA(n_components=variance)
        pca.fit(X)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(explained_variance >= variance) + 1
        return n_components

    def plot2dPCA(self, samples:int = 10000, title:str = None, file_name:str = None):
        X = np.vstack([self.X_train, self.X_val, self.X_test])
        y = np.hstack([self.y_train, self.y_val, self.y_test])

        pca = PCA(n_components=2)
        pca.fit(X)
        X_pca = pca.transform(X[:samples])

        title = title if title else "PCA 2D projection"
        plt.figure(figsize=(8, 6))
        for digit in self.classes:
            plt.scatter(X_pca[y[:samples] == digit, 0], 
                        X_pca[y[:samples] == digit, 1], 
                        label=digit, marker='o', s=10)
        plt.xlabel(r'$\lambda_1$', fontsize=12)
        plt.ylabel(r'$\lambda_2$', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(title='Digit', fontsize=12, loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(file_name) if file_name else plt.show()

    def plot2dTSNE(self, samples:int = 10000, title:str = None, file_name:str = None):
        X = np.vstack([self.X_train, self.X_val, self.X_test])
        y = np.hstack([self.y_train, self.y_val, self.y_test])

        tsne = TSNE(n_components=2, perplexity=30)
        X_tsne = tsne.fit_transform(X[:samples])

        title = title if title else "t-SNE 2D projection"
        plt.figure(figsize=(8, 6))
        for digit in self.classes:
            plt.scatter(X_tsne[y[:samples] == digit, 0], 
                        X_tsne[y[:samples] == digit, 1], 
                        label=digit, marker='o', s=10)
        plt.xticks(ticks=np.linspace(-100, 100, 5))
        plt.xlabel('1st component', fontsize=12)
        plt.ylabel('2nd component', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(title='Digit', fontsize=12, loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(file_name) if file_name else plt.show()

    def plotEigenDigits(self, n_components:int = 10, title:str = None, file_name:str = None):
        X = np.vstack([self.X_train, self.X_val, self.X_test])

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        eigenvecs = pca.components_.reshape(-1, 28, 28)

        title = title if title else "Eigendigits - PCA"
        cols = 5 if n_components >= 5 else n_components
        rows = n_components // cols
        fig, axs = plt.subplots(rows, cols, figsize=(8, 5))
        axs = axs.flatten()
        for i in range(n_components):
            axs[i].imshow(eigenvecs[i], cmap='hot', interpolation='none')
            axs[i].axis('off')
            axs[i].set_title(r'$v_{}$'.format(str({i+1})), fontsize=12)
        fig.suptitle('Eigendigits - PCA', fontsize=16)
        plt.tight_layout()
        plt.savefig(file_name) if file_name else plt.show()