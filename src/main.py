import mnist_loader
import network
import sgd

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

nn = network.Network([784, 15, 10])
nnSGD = sgd.SGD(nn, training_data)
nnSGD.RunTraining()
