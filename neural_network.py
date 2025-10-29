import numpy as np
from utils import sigmoid, sigmoid_derivative

class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.1, seed=42):
        np.random.seed(seed)
        self.lr = lr
        self.sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.W = []
        self.b = []
        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.L):
            in_dim, out_dim = self.sizes[i], self.sizes[i+1]
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.W.append(np.random.uniform(-limit, limit, (in_dim, out_dim)))
            self.b.append(np.zeros((1, out_dim)))

    def forward(self, X):
        a = X
        activations = [a]
        for i in range(self.L):
            z = np.dot(a, self.W[i]) + self.b[i]
            a = sigmoid(z)
            activations.append(a)
        return activations

    def backward(self, activations, y_true):
        grads_W = [None] * self.L
        grads_b = [None] * self.L
        y_pred = activations[-1]
        delta = (y_pred - y_true) * sigmoid_derivative(y_pred)

        for l in reversed(range(self.L)):
            a_prev = activations[l]
            grads_W[l] = np.dot(a_prev.T, delta) / a_prev.shape[0]
            grads_b[l] = np.mean(delta, axis=0, keepdims=True)
            if l > 0:
                delta = np.dot(delta, self.W[l].T) * sigmoid_derivative(activations[l])
        return grads_W, grads_b

    def update(self, grads_W, grads_b):
        for i in range(self.L):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

    def mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def predict(self, X):
        return self.forward(X)[-1]