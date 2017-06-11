import numpy as np
from enum import Enum

class Activations(Enum):
    def sigmoid(self, z, derivate = False):
        if derivate:
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        else:
            return 1. / (1. + np.exp(-z))
    def tanh(self, z, derivate = False):
        if derivate:
            return (1. - np.power(self.tanh(z), 2))
        else:
            return np.tanh(z)
    def relu(self, z, derivate = False):
        if derivate:
            if z < 0:
                return 0.
            else:
                return 1.
        else:
            return max(0., z)