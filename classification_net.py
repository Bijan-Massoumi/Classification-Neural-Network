import numpy as np
import pickle
import weight_layer as wl

class ClassificationNetwork:
    def __init__(self, nodes_layer, activation_function, reg_strength, learning_rate):
        """
        nodes_layer is a list in which each value is the amount of nodes for that particular layer

            i.e. [5, 20, 1], would have an input layer of 5 nodes, hidden layer of 20 nodes,
                and output layer of 1 node.
        

        activation_function is a delegate to an activation function used for forward propagation.

        """
        self.__activation_function = activation_function
        self.__reg_strength = reg_strength
        self.__learning_rate = learning_rate
        self.num_layers = len(nodes_layer)
        self.layers = []
        for i in range(len(nodes_layer)-1):
            self.layers.append(wl.WeightLayer(nodes_layer[i+1], nodes_layer[i]))

    @staticmethod
    def load_network(path, activation_function, reg_strength = 0.01, learning_rate = 0.01):
        """Loads and returns serialized network with given activation function that trained this network."""
        with open(path, "rb") as f:
            layers = pickle.load(f)
        nodes_layer = [layers[0].weights.shape[0]]
        for layer in layers:
            nodes_layer.append(layer.weights.shape[1])
        network = ClassificationNetwork(nodes_layer, activation_function)
        network.layers = layers
        return network

    def serialize_network(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.layers, f)
    
    def train_network(self, images, targets):
        for i in range(images.shape[0]):   
            propagate_backward(images[i,:], targets[i,:])

    def propagate_backward(self, image, target, debug = False):
        """ Propagate error back through network, updating weights """
        activations = self.propagate_forward(image)
        delta = activations[-1] - target
        if debug:
            print(delta)
        for i in reversed(range(len(self.layers))):
            dW = np.dot(activations[i-1].T, delta) + (self.__reg_strength * self.layers[i])
            dB = np.sum(delta, axis=0, keepdims=True)
            self.layers[i].weights -= (self.__learning_rate * dW)
            self.layers[i].biases -= (self.__learning_rate * dB)
            delta = np.dot(delta, self.layers[i].weights.T) * self.__activation_function(\
                                                                activations[i-1], True)

    def get_prediction(self,image):
        return np.around(self.propagate_forward(image_data)[-1])
    
    def propagate_forward(self, image):
        """ Runs network forward and returns activations from layers """
        activation = [image_data]
        for layer in self.layers[:-1]:
            activation.append(self.__activation_function(np.dot(activation[-1] ,\
                                                layer.weights ) + layer.biases))
        #Softmax activation in last layer for best results
        activation.append(softmax(np.dot(activation[-1], self.layers[-1].weights) + \
                                                self.layers[-1].biases))
        return activation
    
    def softmax(z):
        exp_result = np.exp(z)
        return (exp_result / np.sum(exp_result, axis=1, keepdims=True))
    