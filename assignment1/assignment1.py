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

different_measures()
# Method 2 (and 16, which is the same, give the best results.)
# Followed by 3 and 10, which are correlation and minkowski

