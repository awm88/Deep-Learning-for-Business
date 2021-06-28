#!/usr/bin/env python
# coding: utf-8

# In[572]:


pip install autopep8


# In[573]:


#importing libraries and setting up data
import numpy as np
import pandas as pd
fname = 'assign1_data.csv'
data = np.genfromtxt(fname, dtype='float', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]


# In[574]:


#a dense fully connected layer
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """
        Initialize weights & biases.
        Weights should be initialized with values drawn from a normal
        distribution scaled by 0.01.
        Biases are initialized to 0.0.
        """
        np.random.seed(0)
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        """
        A forward pass through the layer to give z.
        Compute it using np.dot(...) and then add the biases.
        """
        self.inputs = inputs
        self.z = np.dot(self.inputs,self.weights) + self.biases
    def backward(self, dz):
        """
        Backward pass
        """
        # Gradients of weights
        self.dweights = np.dot(self.inputs.T, dz)
        # Gradients of biases
        self.dbiases = np.sum(dz, axis=0, keepdims=True)
        # Gradients of inputs
        self.dinputs = np.dot(dz, self.weights.T)


# In[575]:


class ReLu:
    """
    ReLu activation
    """
    def forward(self, z):
        """
        Forward pass
        """
        self.z = z
        self.activity = np.maximum(0,self.z)
    def backward(self, dactivity):
        """
        Backward pass
        """
        self.dz = dactivity.copy()
        self.dz[self.z <= 0] = 0.0


# In[576]:


class Softmax:
    def forward(self, z):
        """
        """
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.probs = e_z / e_z.sum(axis=1, keepdims=True)
        return self.probs
    def backward(self, dprobs):
        """
        """
        # Empty array
        self.dz = np.empty_like(dprobs)
        for i, (prob, dprob) in enumerate(zip(self.probs, dprobs)):
            # flatten to a column vector
            prob = prob.reshape(-1, 1)
            # Jacobian matrix
            jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
            self.dz[i] = np.dot(jacobian, dprob)


# In[577]:


class CrossEntropyLoss:
    def forward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # clip to prevent division by 0
        # clip both sides to not bias up.
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        # negative log likelihoods
        loss = -np.sum(oh_y_true * np.log(probs_clipped), axis=1)
        return loss.mean(axis=0)
    def backward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # Number of examples in batch and number of classes
        batch_sz, n_class = probs.shape
        # get the gradient
        self.dprobs = -oh_y_true / probs
        # normalize the gradient
        self.dprobs = self.dprobs / batch_sz


# In[578]:


class SGD:
    """
    """
    def __init__(self, learning_rate=1.0):
        # Initialize the optimizer with a learning rate
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights = self.learning_rate*layer.dweights
        layer.biases = self.learning_rate*layer.dbiases


# In[579]:


#convert probabilities to predictions
def predictions(probs):
    """
    """
    y_preds = np.argmax(probs, axis=1)
    return y_preds

def accuracy(y_preds, y_true):
    """
    """
    return np.mean(y_preds == y_true)


# In[580]:


def forward_pass(X, y_true,oh_y_true):
    ""
    ""
    dense1.forward(X)
    activation1.forward(dense1.z)
    dense2.forward(activation1.activity)
    activation2.forward(dense2.z)
    dense3.forward(activation2.activity)
    probs = output_activation.forward(dense3.z)
    loss = crossentropy.forward(probs,oh_y_true)
    return probs, loss

    # ### A single backward pass through the entire network.
def backward_pass(probs,y_true,oh_y_true):
    ""
    ""
    crossentropy.backward(probs,oh_y_true)
    output_activation.backward(crossentropy.dprobs)
    dense3.backward(output_activation.dz)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dz)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dz)


# In[581]:


#setting up hyperparameters
input_layer=3
hidden_1 = 4
hidden_2 = 8
batch_sz = 10
epochs = 10
output_layer=3
n_class=3
n_batch = X_train.shape[0] // batch_sz 

dense1 = DenseLayer(input_layer,hidden_1)
activation1 = ReLu()
dense2 = DenseLayer(hidden_1,hidden_2)
activation2 = ReLu() 
dense3 = DenseLayer(hidden_2, output_layer)

output_activation = Softmax()
crossentropy = CrossEntropyLoss()
optimizer = SGD()


# In[582]:


def one_hot_y(n_class,y_true):
    y_true.astype('int')
    one_hot_y = np.eye(n_class)[y_true]
    return one_hot_y


# In[583]:


#training loop
learning_rate=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for lr in learning_rate:
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        for batch_i in range(n_batch):
            # Get batch
            batch_start = batch_sz * batch_i
            # Get a mini-batch of data from X_train and y_train. It should have batch_sz examples.
            batch_end = batch_sz * (batch_i+1)
            x_batch = X_train[ batch_start : batch_end ]
            y_true = y_train[ batch_start : batch_end ]
            oh_y_true=one_hot_y(n_class,y_true)
            # Forward pass
            probs, loss = forward_pass(x_batch, y_true,oh_y_true)
            print(" Batch {}|{} - Epoch {}|{} - Accuracy: {} - Loss {} ".format( batch_i+1, n_batch, 
                                        epoch+1, epochs,
                                        np.round(acc,decimals=3)*100,
                                        np.round(loss,decimals=3)))
            print(f"Accuracy Model: {acc*100}%")
            y_pred = predictions(probs)
            acc = accuracy(y_pred, y_train[batch_start : batch_end])

            #backward pass
            backward_pass(probs,y_true,oh_y_true)

            optimizer.update_params(dense3)
            optimizer.update_params(dense2)
            optimizer.update_params(dense1)


# In[584]:


probs, loss = forward_pass(X_test,y_test,np.eye(n_class)[y_test])   
y_preds=predictions(probs)
print("Testaccuracy:{:.2f}".format(accuracy(y_preds,y_test)))

