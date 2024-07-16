import numpy as np

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
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

# the softmax activation
class activation_softmax:
    def forward(self,inputs):
        #subracting the max to reduce the risk of overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        total_sum = np.sum(exp_values,axis=1,keepdims=True)
        probabilities = exp_values/total_sum
        self.output = probabilities
        ##

