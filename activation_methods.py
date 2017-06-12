import numpy as np
from enum import Enum

class Activations(Enum):
    def sigmoid(z, derivate = False):
        if derivate:
            return z * (1. - z)
        else:
            return 1. / (1. + np.exp(-z))
    def tanh(z, derivate = False):
        if derivate:
            return (1. - np.power(z, 2))
        else:
            return np.tanh(z)
    def relu(z, derivate = False):
        if derivate:
            if z < 0:
                return 0.
            else:
                return 1.
        else:
            return max(0., z)