import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

def one_hot(y, num_classes=10):
    out = np.zeros((y.size, num_classes))
    out[np.arange(y.size), y] = 1
    return out