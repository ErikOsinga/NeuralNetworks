

import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def test_set_MLP_default():
	"""
	Plots the test accuracy (or loss) for the default settings of the MNIST_mlp 
	as is on the KERAS git.
	"""

	testloss = np.load('./test_loss_MLP_default.npy')
	testacc = np.load('./test_accuracy_MLP_default.npy')

	print ('Mean accuracy %4f, standard deviation: %4f' %(np.mean(testacc),np.std(testacc))) 

	plt.title('Test accuracy for 50 different random initializations')
	plt.hist(testacc)
	plt.xlabel('Accuracy')
	plt.ylabel('Count')
	plt.show()

test_set_MLP_default()