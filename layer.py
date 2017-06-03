import numpy as np

class layer:
    def __init__(self,num_nodes, next_layer = None):
        self.num_nodes = num_nodes
        self.activation = np.zeros(shape = (num_nodes,1))
        self.next = next_layer
        
        if next_layer:
            self.theta = np.random.rand(next_layer.num_nodes, self.num_nodes + 1)
            