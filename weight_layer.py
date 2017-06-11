import numpy as np
import activation_methods as am

class WeightLayer:
    """Class to store weights from layers i to j, and biases in layer j."""
    def __init__(self, num_nodes, num_prev_nodes):
        if num_prev_nodes == 0:
            raise ValueError('Invalid number of nodes in previous layer')
        self.velocity_biases = np.zeros((1, num_nodes))
        self.velocity_weights = np.zeros((num_prev_nodes, num_nodes))
        self.biases = np.zeros((1, num_nodes))
        self.weights = np.random.randn(num_prev_nodes, num_nodes) / np.sqrt(num_prev_nodes)

    def nesterov_momentum_weight(self, momentum, learning_rate, gradient):
        """Momentum update for weights"""
        velocity_prev = self.velocity_weights
        self.velocity_weights = momentum * self.velocity_weights - learning_rate * gradient
        return (-momentum * velocity_prev + (1 + momentum) * self.velocity_weights)

    def nesterov_momentum_bias(self, momentum, learning_rate, gradient):
        """Momentum update for biases"""
        velocity_prev = self.velocity_biases
        self.velocity_biases = momentum * self.velocity_biases - learning_rate * gradient
        return (-momentum * velocity_prev + (1 + momentum) * self.velocity_biases)
        
        