import numpy as np 
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances

train_data = np.loadtxt('data/train_in.csv',delimiter=',') # Shape is (1707,256)
train_labels = np.loadtxt('data/train_out.csv') # Shape is (1707,)
test_data = np.loadtxt('data/test_in.csv',delimiter=',')# Shape is (1000,256)
test_labels = np.loadtxt('data/test_out.csv') # Shape is (1000,)
digits = range(0,10)

methods =  ['braycurtis', 'canberra', 'Euclidian',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

def analyze_distance(measure=2):
	"""
	Function to analyze the distance between images and classify the digits
	based on 10 clusters (clouds).
	"""

	all_centers = [] # will be a numpy array containing center 0 until 9, shape (10,256)
	all_radii = []
	all_ni = []
	for digit in digits:
		# find the training data for the current digit
		cloud = train_data[train_labels == digit]
		ni = len(cloud)
		center = np.mean(cloud, axis=0)
		assert center.shape == (256,)

		if measure == 2:
			radius = np.abs(center - train_data) # (1707,256) array containing vector difference
			radius = np.linalg.norm(radius,axis=1) # (1707,) array containing distance to all points
			radius = np.max(radius)
		else:
			radius = pairwise_distances(center.reshape(1,256),train_data,metric=methods[measure])
			radius = np.max(radius)

		all_centers.append(center)
		all_radii.append(radius)
		all_ni.append(ni)

	all_centers = np.asarray(all_centers)
	assert all_centers.shape == (10,256)
	all_radii = np.asarray(all_radii)
	all_ni = np.asarray(all_ni)
	assert all_radii.shape == (10,) == all_ni.shape

	return all_centers, all_radii, all_ni

def classify_distance(all_centers,data,measure):
	digit_distances = [] # (10,1707) array containing the distances for each number 
						# for each training instance
	for digit in digits:
		if measure == 2:
			distance = data - all_centers[digit]
			distance = np.linalg.norm(distance,axis=1)
		else:
			distance = pairwise_distances(data,all_centers[digit].reshape(1,256),metric=methods[measure])

		digit_distances.append(distance)

	digit_distances = np.asarray(digit_distances)
	classify = np.argmin(digit_distances,axis=0) # the digit is the index of the minimum

	assert len(classify) == len(data)

	return classify

def different_measures():
	""" Euclidian """
	measure = 2

	all_centers, all_radii, all_ni = analyze_distance(measure)
	distance_matrix = np.zeros((10,10)) # stores the distances between cloud_i and cloud_j in 
										# position (i,j) or position (j,i)
	for iterable in itertools.combinations(range(0,10),2):
		i,j = iterable
		distance = np.linalg.norm(all_centers[i] - all_centers[j])
		distance_matrix[i,j] = distance
		distance_matrix[j,i] = distance
	"""
	print ('Minimum distance is found between digit 7 and 9:')
	print (distance_matrix[7,9]) 
	# the minimum distance is between 7 and 9. Hardest to seperate.
	# 9 and 4 are the second minimum distance.
	# 0 and 1 are the easiest, but one is the easiest it seems.
	print ('Sum of distances per digit:')
	print (np.sum(distance_matrix,axis=0)) # to see which numbers are most easily classified.
	"""
	
	classify = classify_distance(all_centers,train_data,measure)
	C_train = confusion_matrix(train_labels,classify)
	C_train = C_train / np.sum(C_train,axis=1)
	# print (C_train)
	# Pretty good score on the train data

	classify_test = classify_distance(all_centers,test_data,measure)
	C_test = confusion_matrix(test_labels,classify_test)
	C_test = C_test / np.sum(C_test,axis=1)
	# print (C_test)
	# Pretty ok still, the two and three dropped the most. 
	# for i in range(10):
	# 	print (i, C_train[i][i], C_test[i][i], (C_train[i][i]-C_test[i][i]))

	# We think the difference is because 2 and 3 are more susceptible to being writting in a 
	# different way than for example 0 and 1.


	""" Other metrics """
	max_accuracy = 0
	best_method = 0
	for measure in range(len(methods)): # use all methods and see which gives highest accuracy
		all_centers, all_radii, all_ni = analyze_distance(measure)
		classify_test = classify_distance(all_centers,test_data,measure)
		C_test = confusion_matrix(test_labels,classify_test)
		C_test = C_test / np.sum(C_test,axis=1)
		result_accuracy = 0
		for i in range(10):
			result_accuracy += C_test[i][i]
		result_accuracy /= 10.

		print (measure, result_accuracy)
		if result_accuracy > max_accuracy:
			max_accuracy = result_accuracy
			best_method = measure

	print ('Best method, accuracy:')
	print (best_method, max_accuracy)

# different_measures()
# Method 2 (and 16, which is the same, give the best results.)
# Followed by 3 and 10, which are correlation and minkowski

def extra_feature(plot=False):
	# Use an extra feature to discriminate between 5 and 7.
	# We'll use the number of pixels, that are written on. Just sum it all up.
	digit5 = train_data[train_labels == 5]
	digit7 = train_data[train_labels == 7]

	feature5 = np.sum(digit5,axis=1)
	feature7 = np.sum(digit7,axis=1)

	# data = np.append(digit5,digit7,axis=0)
	# feature = np.sum(data,axis=1)

	# plt.hist(feature)
	# plt.show()

	plt.title('Histogram of the feature for digit 5 and 7')
	binedges = np.array([-217.014 , -200.4061, -183.7982, -167.1903, -150.5824, -133.9745,
       -117.3666, -100.7587,  -84.1508,  -67.5429,  -50.935, -34, -17, 0, 20, 40])
	n5, bins5, patches5 = plt.hist(feature5,label='digit 5',alpha=0.5,bins=binedges)
	n7, bins7, patches7 = plt.hist(feature7,label='digit 7',alpha=0.5,bins=binedges)
	plt.legend()
	plt.xlabel('Feature')
	plt.ylabel('Count')
	if plot	:
		plt.show()
	else:
		plt.close()

	PX_C5 = n5 / np.sum(n5)
	PC5 = np.sum(n5) / (np.sum(n5) + np.sum(n7))
	PX_C7 = n7 / np.sum(n7)
	PC7 = np.sum(n7) / (np.sum(n5) + np.sum(n7))
	PX = (n5+n7) / (np.sum(n5+n7))

	print (np.sum(PX_C5))
	print (np.sum(PX_C7))
	print (np.sum(PC5))
	print (np.sum(PC7))
	# Probability that the class is C, when X in the current bin.
	PC5_X = (PX_C5 * PC5) / PX
	PC7_X = (PX_C7 * PC7) / PX

	print (PC5_X, PC7_X)

	return PC5_X, PC7_X, bins5, bins7

PC5_X, PC7_X, bins5, bins7 = extra_feature(True)

def bayes_classification(PC5_X, PC7_X, bins5, bins7):
	digit5 = test_data[test_labels == 5]
	digit7 = test_data[test_labels == 7]

	data = np.append(digit5,digit7,axis=0)
	feature = np.sum(data,axis=1)

	bincenters5 = 0.5*(bins5[1:]+bins5[:-1])
	bincenters7 = 0.5*(bins7[1:]+bins7[:-1])

	bins = 15
	# calculate the distance to the center of each bin
	d_to_center5 = np.abs((feature.reshape(len(feature),1) - bincenters5.reshape(1,bins))).T
	d_to_center7 = np.abs((feature.reshape(len(feature),1) - bincenters7.reshape(1,bins))).T

	# find the index of the closest bin to the current feature value
	index_bin5 = np.argmin(d_to_center5,axis=0)
	index_bin7 = np.argmin(d_to_center7,axis=0)

	prob5 = PC5_X[index_bin5]
	prob7 = PC7_X[index_bin7]

	# print (prob5)
	# print ('\n')
	# print (prob7)
	print (prob5 > prob7)

	'''
	all_prob5 = []
	for case in feature5:
		index_bin = np.where(case > bins5)[0][-1] # finds the index of the bin
													# that our feature is in
		prob5 = (PC5_X[index_bin])
		all_prob5.append(prob5)
	all_prob7 = []
	for case in feature7:
		print (np.where(case > bins7))
		print (bins7)
		print (case)
		index_bin = np.where(case > bins7)[0][-1]
		prob7 = (PC7_X[index_bin])
		all_prob7.append(prob7)

	print (all_prob5, all_prob7)
	'''

bayes_classification(PC5_X, PC7_X, bins5, bins7)