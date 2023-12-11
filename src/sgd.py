import random

import numpy as np

from network import Network


class SGD:
    def __init__(self, network, training_data, test_data=None):
        if not isinstance(network, Network):
            raise TypeError("The 'network' parameter must be an instance of the Network class.")

        self.network = network
        self.training_data = list(training_data)
        self.test_data = test_data

        # default values
        self.eta = 0.5
        self.epochs = 100
        self.batch_size = 10

    def set_learning_rate(self, eta):
        self.eta = eta

    def set_epochs_quantity(self, epochs_qty):
        self.epochs = epochs_qty

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def train(self):
        length_test = 0
        if self.test_data:
            length_test = len(self.test_data)

        length = len(self.training_data)
        for i in range(self.epochs):
            random.shuffle(self.training_data)
            batches = [self.training_data[j:j + self.batch_size] for j in range(0, length, self.batch_size)]

            for batch in range(batches):
                self.update_batch(batch)

            if self.test_data:
                print("Epoch {} : {} / {}".format(i, self.evaluate(self.test_data), length_test))
            else:
                print("Epoch {} complete".format(i))

    def update_batch(self, batch):
        nabla_b = [np.zeros(b) for b in self.network.biases]
        nabla_w = [np.zeros(w) for w in self.network.weights]

        
