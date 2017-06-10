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
        self.num_layers = len(nodes_layer)
        self.layers = []
        for i in range(len(nodes_layer)-1):
            self.layers.append(wl.WeightLayer(nodes_layer[i+1], nodes_layer[i]))
    
    def train_network(self, image_data, labels):
        for x,y in zip(image_data,labels):   
            grad_b, grad_w = self.propagate_backward(x, y)

    def get_prediction(self,image_data):
        return np.around(self.__propagate_forward(image_data)[-1])
    
    def propagate_forward(self, image_data):
        activation = [image_data]
        for layer in self.layers:
            activation.append(self.__activation_function(np.dot(layer.weights ,\
                                                activation[-1]) + layer.biases))
        return activation

    def propagate_backward(self, images, targets):
        weight_deltas = []
        bias_deltas = []
        for layer in self.layers:
            weight_deltas.append(np.zeros(shape = layer.weights.shape))
            bias_deltas.append(np.zeros(shape = layer.biases.shape))
            
        
        activations = self.propagate_forward(image)
        bias_deltas = (activations[-1] - targets)
        
        #compute errors for each layer the loop condition revserses the list and excludes the ends of it
        for i in range(num_activations-2,0,-1): 
            bias_deltas[i]  = self.layers[i].weights.transpose().dot(bias_deltas[i+1])\
                * (activations[i] * (1 - activations[i]))
           
        for i, delta in enumerate(total_deltas): #accumulate
            total_deltas[i] = delta + bias_deltas[i+1].dot(activations[i].transpose())
                
        grad = list(map(lambda x: x * (1/len(images)), delta_totals ))          
        return grad
