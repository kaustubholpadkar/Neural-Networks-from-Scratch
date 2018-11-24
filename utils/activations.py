import numpy as np


def sigmoid(x, derivative=False):
    """
    Sigmoid activation function
    :param x: input
    :param derivative: bool - True if derivative of sigmoid function is needed
    :return: sigmoid or derivative of sigmoid for given input
    """
    sig = 1. / (1. + np.exp(-x))
    if derivative:
        return sig * (1. - sig)
    return sig
