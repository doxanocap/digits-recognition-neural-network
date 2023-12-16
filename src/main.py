import mnist_loader
import network
import rest
import logger

if __name__ == '__main__':
    logger.initLogger()
    nn = network.Network([784, 30, 10])
    rest.initREST(nn)

