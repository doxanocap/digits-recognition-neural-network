import numpy as np
import sgd


def weighted_sum(w, a, b):
    return np.dot(w, a) + b


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for y, x in zip(sizes[:-1], sizes[1:])]
        self.SGD = sgd.SGD

    def feed_forward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(weighted_sum(w, a, b))
        return a
