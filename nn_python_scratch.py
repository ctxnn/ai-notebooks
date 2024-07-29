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

# the Loss Class
class Loss:
    #calculation of the data
    def calculate(self,output,y):
        #sample losses
        sample_losses = self.forward(output,y)

        #mean
        data_loss = np.mean(sample_losses)

        #return loss
        return data_loss

class Loss_CategorialCrossEntropy(Loss):
    #forward pass
    def forward(self,y_pred,y_true):
        samples = len(y_pred)

        #prevent data to prevent division by 0
        #the log(0) case
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #for sparse labels
        if len(y_true.shape) == 1:
            correct_cofidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_cofidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_loss_likelihoods = -np.log(correct_cofidences)
        return negative_loss_likelihoods
