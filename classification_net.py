import numpy as np
import preprocess

class ClassificationNetwork:
    def __init__(self,num_inputs,num_classes,num_hidden_layers,nodes_per_layer):
        self.num_layers = num_hidden_layers
        self.weights = allocate_inital_weights(num_hidden_layers,\
                                            num_inputs,num_classes)

    def forward_propagation(self,image_data):
        activations = []
        activations.append(np.concatenate((np.array([1]),image_data)))
        for i in range(0,self.num_layers+1):
            curr_layer = activations[-1].transpose()
            z = self.weights[i].dot(curr_layer)
            activations.append(sigmoid_function(z))
        return activations


    def sigmoid_function(z):
        return 1. / (1. + np.exp(-z))

    def allocate_inital_weights(num_hidden_layers,num_inputs,num_classes):
        weights = []
        for i in range(0,num_layers):
            if i == 0:
                weights.append(np.random.randn(nodes_per_layer,\
                                                    num_inputs + 1))
            else:
                weights.append(np.random.randn(nodes_per_layer,\
                                                nodes_per_layer + 1))
        self.weights.append(np.random.randn(num_classes,nodes_per_layer + 1))

        return weights
