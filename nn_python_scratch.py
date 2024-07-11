import numpy as np
import torch

np.random.seed(0)

class layer_dense:
    # initializing the layer
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros(1,n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights)+self.biases

# the ReLU activaion function
class activation_relu:
    def forward(self,)
        

