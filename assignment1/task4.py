import numpy as np
from matplotlib import pyplot as plt

train_data = np.loadtxt('data/train_in.csv',delimiter=',') # Shape is (1707,256)
train_labels = np.loadtxt('data/train_out.csv') # Shape is (1707,)
test_data = np.loadtxt('data/test_in.csv',delimiter=',')# Shape is (1000,256)
test_labels = np.loadtxt('data/test_out.csv') # Shape is (1000,)
digits = range(0,10)

# make sure X is in shape (256, number of examples)
train_data = train_data.T 
test_data = test_data.T

# Add the bias node
train_data = np.append(train_data,[np.ones(train_data.shape[1])],axis=0) 
test_data = np.append(test_data, [np.ones(test_data.shape[1])],axis=0)

def train_single_layer_perceptron(learning_rate = 0.05):
	"""
	Train on the training set untill 100 % accurate. Return the weights 
	"""

	w = np.random.rand(257,10)  # 256 + 1 for the bias, 10 output nodes.

	bad_class = 1e6 # initialization
	while bad_class > 0:
		bad_class = 0

		for i in range(train_data.shape[1]):
			X = train_data[:,i]
			X = X.reshape(257,1)
			a = np.dot(w.T,X) 
			yhat = np.argmax(a)
			
			ylabel = int(train_labels[i]) # the digit that it should be

			if yhat == ylabel:
				pass
			else:
				# Increase the weights were too low
				w[:,ylabel] += train_data[:,i] * learning_rate
				# Decrease the weights that were too high.
				w[:,yhat] -= train_data[:,i] * learning_rate
				bad_class += 1

		# print ('Number of wrong classifications: ', bad_class)

	return w

w = train_single_layer_perceptron()

def test_single_layer_perceptron(w):
	"""
	Test given weights w on the test set. Print the accuracy.
	"""
	
	bad_class = 0
	for i in range(test_data.shape[1]):
		X = test_data[:,i]
		X = X.reshape(257,1)
		a = np.dot(w.T,X) 
		yhat = np.argmax(a)
		
		ylabel = int(test_labels[i]) # the digit that it should be

		if yhat != ylabel:
			bad_class += 1

	print ('Number of wrong classifications on the test set: ', bad_class)
	print ('Accuracy: ', 1 - bad_class/test_data.shape[1])

test_single_layer_perceptron(w)