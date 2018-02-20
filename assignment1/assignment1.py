import numpy as np 
from matplotlib import pyplot as plt

def analyze_distance():
	"""
	Function to analyze the distance between images and classify the digits
	based on 10 clusters (clouds).
	"""

	train_data = np.loadtxt('data/train_in.csv',delimiter=',') # Shape is (1707,256)
	train_labels = np.loadtxt('data/train_out.csv') # Shape is (1707,)

	digits = range(0,10)
	all_centers = [] # will be a numpy array containing center 0 until 9, shape (10,256)
	all_radii = []
	for digit in digits:
		# find the training data for the current digit
		cloud = train_data[train_labels == digit]
		center = np.mean(cloud, axis=0)
		assert center.shape == (256,)
		all_centers.append(center)

		radius = np.abs(center - train_data) # (1707,256) array containing vector difference
		radius = np.linalg.norm(radius,axis=1) # (1707,) array containing distance to all points
		radius = np.max(radius)
		all_radii.append(radius)

	all_centers = np.asarray(all_centers)
	assert all_centers.shape == (10,256)
	all_radii = np.asarray(all_radii)
	assert all_radii.shape == (10,)



analyze_distance()



