import numpy as np
from matplotlib import pyplot as plt

train_data = np.loadtxt('data/train_in.csv',delimiter=',') # Shape is (1707,256)
train_labels = np.loadtxt('data/train_out.csv') # Shape is (1707,)
test_data = np.loadtxt('data/test_in.csv',delimiter=',')# Shape is (1000,256)
test_labels = np.loadtxt('data/test_out.csv') # Shape is (1000,)
digits = range(0,10)

train_data = train_data.T # make sure X is in shape (256, number of examples)
test_data = test_data.T


# Make new labels to quickly use gradient descent on every node.
Y = []
for i in train_labels:
	temp = np.zeros(10)
	temp[int(i)] = 1
	Y.append(temp)

Y = np.asarray(Y).T 
# should be shape (10, number of examples)
assert Y.shape == (10,train_data.shape[1])


def sigmoid(X):
	"""
	Calculate the sigmoid function for vector X
 	"""

	A = 1/(1+np.exp(-X))

	return A

def digits_net(data, w1, w2):
	"""

	"""

	# add the bias to the training data  shape (257, number of examples)
	X1 = np.append(data,[np.ones(data.shape[1])],axis=0)

	# forward pass to hidden layer 1
	A1 = sigmoid( np.dot(w1.T,X1) ) # shape (10,number of examples)
	
	assert A1.shape == (10,data.shape[1])
	
	A1 = np.append(A1,[np.ones(data.shape[1])],axis=0) # add the bias node again

	# print ('A1: ', A1[:,0])

	# forward pass to output layer
	A2 = np.squeeze(sigmoid( np.dot(w2.T,A1) )) # shape (10, number of examples)
	
	# print ('A2: ', A2[:,0])

	yhat = np.argmax(A2,axis=0) # shape (number of examples,)

	yhat = yhat.reshape(1,data.shape[1]) # reshape to (1, number of examples)

	assert yhat.shape == (1, data.shape[1])

	return A2, yhat

def mse(data, w1, w2):
	'''
	Calculate the MSE for every one of the output nodes and return this as a (10,) array
	'''
	
	A2, yhat = digits_net(data, w1, w2)

	MSE = 1./data.shape[1] * np.sum( (Y - np.asarray(A2) )**2,axis=1)

	# print ('Mean squared error: ', MSE) # should be (10,) 

	return np.sum(MSE), yhat # calculate the mean squared error over all output nodes

def grdmse(data, w1, w2):
	eta = 1e-8

	dw1 = np.copy(w1) # we change every value to dw1 anyways

	# update every single weight value to calculate the derivative..
	for i in range(w1.shape[0]):
		for j in range(w1.shape[1]):
			new_w1 = np.copy(w1)
			new_w1[i][j] = new_w1[i][j]+eta
		
			dw_ij = (mse(data,new_w1, w2)[0] - mse(data, w1, w2)[0]) / eta
			dw1[i][j] = dw_ij

	dw2 = np.copy(w2) # we change every value to dw2 anyways
	for i in range(w2.shape[0]):
		for j in range(w2.shape[1]):
			new_w2 = np.copy(w2)
			new_w2[i][j] = new_w2[i][j] + eta

			dw_ij = (mse(data, w1, new_w2)[0] - mse(data, w1, w2)[0]) / eta
			dw2[i][j] = dw_ij

	MSE, yhat = mse(data, w1, w2)
	print ('Mean squared error: %f' % MSE)
	number_wrong = np.count_nonzero(yhat - train_labels)
	print ('Number of wrongly classified digits: %f ' % number_wrong )

	return dw1, dw2, MSE, number_wrong

def gradient_descent(learning_rate=8.0):
	w1 = np.random.randn(257,10)
	w2 = np.random.randn(11,10)

	print ('Learning rate %f' %learning_rate)

	# dw1, dw2 = grdmse(train_data, w1, w2)

	all_MSE = []
	all_number_wrong = []	
	for epoch in range(4000):#3000):
		print ('\nEpoch number %i ' % epoch)
		dw1, dw2, MSE, number_wrong = grdmse(train_data, w1, w2)
		w1 -= learning_rate * dw1
		w2 -= learning_rate * dw2
		print ('Summed value of the weights w2: %f ' % np.sum(w2) )
		all_MSE.append(MSE)
		all_number_wrong.append(number_wrong)

	np.save('./results/w1_LR_%.2f'%learning_rate,w1)
	np.save('./results/w2_LR_%.2f'%learning_rate,w2)
	np.save('./results/all_MSE_%.2f'%learning_rate,all_MSE)
	np.save('./results/all_number_wrong_%.2f'%learning_rate,all_number_wrong)


gradient_descent()


'''
Mean squared error: 15.764254
Wrong classified digits: 63.000000 
Iterations 1617

'''