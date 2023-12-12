import random

import numpy as np

import network


class SGD:
    def __init__(self, nn, training_data, test_data=None):
        if not isinstance(nn, network.Network):
            raise TypeError("The 'network' parameter must be an instance of the Network class.")

        self.network = nn
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

    def RunTraining(self):
        length_test = 0
        if self.test_data:
            length_test = len(self.test_data)

        length = len(self.training_data)
        for i in range(2):
            random.shuffle(self.training_data)
            batches = [
                self.training_data[j:j + self.batch_size]
                for j in range(0, length, self.batch_size)]

            for batch in batches:
                self.update_batch(batch)

            if self.test_data:
                print("Epoch {} : {} / {}".
                      format(i, self.evaluate(self.test_data), length_test))
            else:
                print("Epoch {} complete".
                      format(i))

        self.network.save_params()

    def update_batch(self, batch):
        delta_ws = [np.zeros(w.shape) for w in self.network.weights]
        delta_bs = [np.zeros(b.shape) for b in self.network.biases]

        # x is an input array, which represented as np.ndarray of len 784
        # y is an output array, of len 10 with desired results
        for x, y in batch:
            # main logic starts here, with the algorithm called back propagation
            delta_w_diffs, delta_b_diffs = self.backprop(x, y)
            delta_ws = [nw + dwd for nw, dwd in zip(delta_ws, delta_w_diffs)]
            delta_bs = [nw + dbd for nw, dbd in zip(delta_bs, delta_b_diffs)]
        self.network.weights = [
            w - (self.eta / self.batch_size) * dw
            for w, dw in zip(self.network.weights, delta_ws)]
        self.network.biases = [
            b - (self.eta / self.batch_size) * nb
            for b, nb in zip(self.network.biases, delta_bs)]

    def backprop(self, x, y):
        delta_ws = [np.zeros(w.shape) for w in self.network.weights]
        delta_bs = [np.zeros(b.shape) for b in self.network.biases]

        # array of values of z
        # where z(l) = w(l) * a^(l-1) + b^(l)
        weighted_sums = []

        # x = activation of l1 - layer 1
        activation = x
        # list of all activations
        activations = [x]
        for w, b in zip(self.network.weights, self.network.biases):
            z = np.dot(w, activation) + b
            weighted_sums.append(z)
            activation = network.sigmoid(z)
            activations.append(activation)

        cost_delta = 2 * (activations[-1] - y) * network.sigmoid_prime(weighted_sums[-1])
        delta_bs[-1] = cost_delta
        delta_ws[-1] = np.dot(cost_delta, activations[-2].transpose())

        num_layers = len(self.network.sizes)
        for i in range(2, num_layers):
            delta_bs[-i] = network.compute_delta_b(self.network.weights[-i+1], cost_delta, activations[-i])
            cost_delta = delta_bs[-i]
            delta_ws[-i] = network.compute_delta_w(cost_delta, activations[-i-1])

        return delta_ws, delta_bs

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.network.feed_forward(x)), y)
            for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
