from src import network

network = network.Network([784, 30, 10])

sgd = network.SGD(network, [])

sgd.train()
