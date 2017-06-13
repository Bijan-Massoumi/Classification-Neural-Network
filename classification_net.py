import numpy as np
import pickle
import weight_layer as wl

class ClassificationNetwork:
    def __init__(self, nodes_layer = [], activation_function = None):
        """
        nodes_layer is a list in which each value is the amount of nodes for that particular layer

            i.e. [5, 20, 1], would have an input layer of 5 nodes, hidden layer of 20 nodes,
                and output layer of 1 node.


        activation_function is a delegate to an activation function used for forward propagation.

        """
        self.__activation_function = activation_function
        self.layers = []
        for i in range(len(nodes_layer)-1):
            self.layers.append(wl.WeightLayer(nodes_layer[i+1], nodes_layer[i]))

    def load_network(self, path, activation_function):
        """Loads and returns serialized network, requires given activation function that trained this network."""
        with open(path, "rb") as f:
            layers = pickle.load(f)
        self.layers = layers
        self.__activation_function = activation_function

    def serialize_network(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.layers, f)

    def train_network(self, images, targets, epochs, reg_strength, learning_rate, batch_size, momentum, debug):
        for epoch in range(epochs):
            sample = np.random.permutation(images.shape[0])
            for i in range(images.shape[0] // batch_size):
                index = sample[(i*batch_size):((i*batch_size)+batch_size)]  # Get batch_size random samples
                self.propagate_backward(images[index,:], targets[index,:], reg_strength,\
                                        learning_rate, momentum, debug)
            if (images.shape[0] % batch_size) != 0: # Feed remaining samples through network
                index = sample[-(images.shape[0] % batch_size):]
                self.propagate_backward(images[index,:], targets[index,:], reg_strength,\
                                        learning_rate, momentum, debug)

    def propagate_backward(self, image_batch, target, reg_strength, learning_rate, momentum, debug):
        """ Propagate error back through network, updating weights """
        activations = self.propagate_forward(image_batch)
        delta = activations[-1] - target
        if debug:
            print(sum(delta))
        for i in reversed(range(len(self.layers))):
            dW = np.dot(activations[i].T, delta) - (reg_strength * self.layers[i].weights)
            dB = np.sum(delta, axis=0, keepdims=True)
            self.layers[i].weights += self.layers[i].nesterov_momentum_weight(momentum, learning_rate, dW)
            self.layers[i].biases += self.layers[i].nesterov_momentum_bias(momentum, learning_rate, dB)
            delta = np.dot(delta, self.layers[i].weights.T) * self.__activation_function(\
                                                                activations[i], True)

    def compute_accuracy(self, test_images, test_targets):
        correct = 0
        for i in range(test_images.shape[0]):
            fwd = self.get_prediction(test_images[i,:])[0]
            prediction = np.zeros_like(fwd, dtype=np.float)
            prediction[np.where(fwd==np.max(fwd))] = 1
            correct += np.array_equal(prediction, test_targets[i,:]) * 1.
        return correct/test_images.shape[0]

    def get_prediction(self,image):
        return self.propagate_forward(image)[-1]

    def propagate_forward(self, image_batch):
        """ Runs network forward and returns activations from layers """
        activation = [image_batch]
        for layer in self.layers[:-1]:
            activation.append(self.__activation_function(np.dot(activation[-1] ,\
                                                layer.weights) + layer.biases))
        #Softmax activation in last layer for best results
        activation.append(self.softmax(np.dot(activation[-1], self.layers[-1].weights) + \
                                                self.layers[-1].biases))
        return activation

    def softmax(self, z):
        exp_result = np.exp(z)
        return (exp_result / np.sum(exp_result, axis=1, keepdims=True))
