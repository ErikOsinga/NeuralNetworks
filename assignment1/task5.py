import numpy as np
from matplotlib import pyplot as plt

train_data = np.loadtxt('data/train_in.csv',delimiter=',') # Shape is (1707,256)
train_labels = np.loadtxt('data/train_out.csv') # Shape is (1707,)
test_data = np.loadtxt('data/test_in.csv',delimiter=',')# Shape is (1000,256)
test_labels = np.loadtxt('data/test_out.csv') # Shape is (1000,)
digits = range(0,10)

train_data = train_data.T # make sure X is in shape (256, number of examples)
test_data = test_data.T

def sigmoid(X):
	"""
	Calculate the sigmoid function for vector X
 	"""

	A = 1/(1+np.exp(-X))

	return A


def xor_net(x1,x2,weights):
	"""
	weights is the vector [w1, ... , w9] containing the weights. 
	"""

	# reshape weights for matrix multiplications
	w1 = weights[:6].reshape(3,2)
	w2 = weights[6:].reshape(3,1)

	X1 = np.array([+1,x1,x2]).reshape(3,1)
	# forward pass to hidden layer 1
	A1 = sigmoid( np.dot(w1.T,X1) ) # shape (2,1)
	assert A1.shape == (2,1)
	A1 = np.append(A1,[[+1]],axis=0) # add the bias node

	A2 = np.squeeze(sigmoid( np.dot(w2.T,A1) )) # shape (1,1)

	if A2 > 0.5:
		yhat = 1
	else:
		yhat = 0

	return A2, yhat

def mse(weights):
	input_vectors = [ [0,0], [0,1], [1,0], [1,1] ]
	targets = [0, 1, 1, 0]

	predictions = [] # numeric
	yhats = [] # 0 or 1
	for x1,x2 in input_vectors:
		A2, yhat = xor_net(x1,x2,weights)
		predictions.append(A2)
		yhats.append(yhat)

	wrong_predictions = np.sum(np.abs( np.asarray(targets) - np.asarray(yhats)))

	MSE = 1./4 * np.sum( (np.asarray(targets) - np.asarray(predictions) )**2 )

	print ('Mean squared error: ', MSE)

	print ('Wrong predictions: ', wrong_predictions)

	return MSE

def grdmse(weights):
	eta = 1e-3

	dw = []
	for i in range(len(weights)):
		new_weights = np.copy(weights)
		new_weights[i] = new_weights[i]+eta
		
		dw_i = (mse(new_weights) - mse(weights)) / eta
		dw.append(dw_i)

	return np.asarray(dw)


def gradient_descent(learning_rate=1):
	weights = np.random.randn(9)

	for i in range(3000):
		weights = weights - learning_rate * grdmse(weights)

		
gradient_descent()