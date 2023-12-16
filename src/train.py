import mnist_loader
import network
import sgd

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    nn = network.Network([784, 30, 10])

    nnSGD = sgd.SGD(nn, training_data, test_data=test_data)
    nnSGD.set_epochs_quantity(200)
    nnSGD.set_learning_rate(3.0)
    nnSGD.set_batch_size(10)
    nnSGD.run_training()
