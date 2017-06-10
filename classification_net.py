import numpy as np
import preprocess
import weight_layer as wl

class ClassificationNetwork:
    def __init__(self, nodes_layer, activation_function):
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
    
    def train_network(self, image_data, labels):
        prop_fwd = self.propagate_forward(image_data)
        self.propagate_backward(prop_fwd, labels)

    def get_prediction(self,image_data):
        return np.around(self.__propagate_forward(image_data)[-1])
    
    def propagate_forward(self, image_data):
        activation = [image_data]
        for layer in self.layers:
            # adding 1 for bias compuation 1 * bias collumn in the dot compuation
            print(activation[-1].shape)
            current_a = np.concatenate([np.array([1]), activation[-1]])
            activation.append(self.__activation_function(layer.weights.dot(current_a)))
        return activation

    def propagate_backward(self, predictions, labels):
        print(None)
