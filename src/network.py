import os

import numpy as np
import json


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


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
            a = sigmoid(np.dot(w, a) + b)
        return a

    def save_params(self):
        current_params = {
            'sizes': self.sizes,
            'biases': [item.tolist() if isinstance(item, np.ndarray) else item for item in self.biases],
            'weights': [item.tolist() if isinstance(item, np.ndarray) else item for item in self.weights]
        }

        if os.path.exists(self.storage_path):
            file = open(self.storage_path, 'r')
            file_content = file.read()
            file.close()

            if file_content != "" and len(file_content) != 0:
                self.storage = json.loads(file_content)
                for case in self.storage:
                    if case['sizes'] == self.sizes:
                        case['biases'] = current_params['biases']
                        case['weights'] = current_params['weights']
                return

        file = open(self.storage_path, 'w')
        json.dump([current_params], file, sort_keys=True, indent=4)
        file.close()

    def load_params(self):
        if os.path.exists(self.storage_path):
            file = open(self.storage_path, 'r+')
            file_content = file.read()
            file.close()

            if file_content != "" and len(file_content) != 0:
                self.storage = json.loads(file_content)
                for case in self.storage:
                    if case['sizes'] == self.sizes:
                        self.biases = [np.array(item) for item in case['biases']]
                        self.weights = [np.array(item) for item in case['weights']]
                return

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feed_forward(x)), y)
            for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)
