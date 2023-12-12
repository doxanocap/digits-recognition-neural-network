import numpy as np
import json


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def compute_delta_b(a, a_prev, z):
    return 2 * np.dot(a.transpose(), a_prev) * sigmoid_prime(z)


def compute_delta_w(delta_b, a_prev):
    return np.dot(delta_b, a_prev.transpose())


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.storage = {}
        self.biases = []
        self.weights = []

        self.storage_path = "../data/storage.json"
        self.load_params()

    def feed_forward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w.transpose(), a) + b)
        return a

    def save_params(self):
        params = {
            'sizes': self.sizes,
            'biases': [b.tolist() for b in self.biases],
            'weights':  [w.tolist() for w in self.weights]
        }
        file = open(self.storage_path, 'r+')

        data = []
        file_content = file.read()
        if file_content != "":
            data = json.loads(file.read())
        data.append(params)
        json.dump(data, file)

    def load_params(self):
        with open(self.storage_path) as file:
            file_content = file.read()
            if file_content != "":
                self.storage = json.loads(file_content)

                for param in self.storage:
                    if param['sizes'] == self.sizes:
                        self.biases = np.array(param['biases'])
                        self.weights = np.array(param['weights'])
                        return

            self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
