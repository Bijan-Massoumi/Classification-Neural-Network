from layer import *

class ClassificationNetwork:
    def __init__(self,num_inputs,num_classes,num_hidden_layers,nodes_per_layer):
        self.weights = allocate_inital_weights(num_hidden_layers,\
                                            num_inputs,num_classes)



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
