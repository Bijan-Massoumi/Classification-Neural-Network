import numpy as np
from enum import Enum

class Activations(Enum):
    def sigmoid(z):
        return 1. / (1. + np.exp(-z))
    def tanh(z):
        return np.tanh(z)
    def relu(z):
        return max(0, z)
