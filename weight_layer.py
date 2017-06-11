import numpy as np
import activation_methods as am

class WeightLayer:
    """Class to store weights from layers i to j, and biases in layer j."""
    def __init__(self, num_nodes, num_prev_nodes):
        if num_prev_nodes == 0:
            raise ValueError('Invalid number of nodes in previous layer')
        self.velocity = np.zeros((num_prev_nodes, num_nodes))
        self.biases = np.zeros((1, num_nodes))
        self.weights = np.random.randn(num_prev_nodes, num_nodes) / np.sqrt(num_prev_nodes)

    def nesterov_momentum(self, momentum, learning_rate, gradient):
        velocity_prev = self.velocity
        self.velocity = momentum * self.velocity - learning_rate * gradient
        return (-momentum * velocity_prev + (1 + momentum) * self.velocity)
        
        