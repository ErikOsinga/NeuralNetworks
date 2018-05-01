import numpy as np
from urllib import request
import sys

# For running on local pc
data_dir = '/home/erik/Desktop/vakken/NeuralNetworks/maps_data/'

# for running on Duranium
data_dir = '/data/s1546449/maps_data/'

def generate_link(size,lat,lon,zoom,maptype='roadmap'):
	'''
	Function to generate a link to a certain google maps cutout.

	size --> integer which determines the amount of pixels in a squared image
	lat --> latitude of the center (float) #(Up and Down)
	lon --> longitude of the center (float) #(Left and Right)
	zoom --> zoom factor, integer
	maptype --> 'roadmap' or 'satellite'

	'''
	begin = "http://maps.google.com/maps/api/staticmap?sensor=false"
	end = "&style=feature:all|element:labels|visibility:off"
	key = "&key=AIzaSyD7o6a-MKAtFX08sqHG-_Vkk8OShV6oJmY"
	
	size = str(size)+'x'+str(size)
	center = str(lat)+','+str(lon)
	zoom = str(zoom)

	if maptype not in ['roadmap' , 'satellite']:
		raise ValueError("Wrong maptype")

	link = begin+'&size=%s&center=%s&zoom=%s'%(size,center,zoom)+end+'&maptype=%s'%maptype+key

	return link

def download_from_link(link,filename):
	"""
	Saves PNG images from a link
	"""

	request.urlretrieve(link,data_dir+filename+'.png')

def download_images(number_of_images,num_already_downloaded):
	'''
	Download a square root-able number of images 
	'''

	if int(number_of_images**0.5)**2 != number_of_images:
		raise ValueError("Please give number of images that can be sq. rooted")

	# Number of images per direction 
	num_im = int(number_of_images**0.5) 

	starting_lat = 52.1688731
	starting_lon = 4.4569086

	# For zoom 15 and size 512x512 the increments must be 
	increment_lat = 0.003200
	increment_lon = 0.005450

	# Because first 2500 images have already been downloaded:
	starting_lat = starting_lat + num_already_downloaded * increment_lat
	starting_lon = starting_lon + num_already_downloaded * increment_lon

	# Walk from starting_coords to ending_coords
	ending_lat = starting_lat + num_im * increment_lat
	ending_lon = starting_lon + num_im * increment_lon

	print ('Downloading %i images, starting at image number %i'%(number_of_images
                                                ,num_already_downloaded+1) )
	i = num_already_downloaded + 1
	for lat in np.linspace(starting_lat,ending_lat,num_im):
		for lon in np.linspace(starting_lon,ending_lon,num_im):
			image_number = '%06i' % i
			
			link = generate_link(size=512,lat=lat,lon=lon,zoom=15,maptype='satellite')
			
			filename = image_number+'_satellite'
			download_from_link(link,filename)

			link = generate_link(size=512,lat=lat,lon=lon,zoom=15,maptype='roadmap')
			
			filename = image_number + '_roadmap'
			download_from_link(link,filename)
			
			sys.stdout.write("\r%i"%i)
			sys.stdout.flush()

			i += 1



# Definining a square in which the data must fit
left_lat = 52.133903
left_lon = 4.309731

right_lat = 52.081739
right_lon =  7.046543

upper_lat = 53.507297
upper_lon = 6.410731

lower_lat = 51.297145
lower_lon = 5.398244
# Each image is about 500m

number_of_images = 6400
download_images(number_of_images,num_already_downloaded=2500)
