from matplotlib import pyplot as io
import numpy as np
import numpy.matlib as nm
import random
import scipy.misc

# number of colors used for different images
K = [2, 5, 10, 15, 20]
max_iterations = 50
filename = ['Koala.jpg', 'Penguins.jpg']

def closest_centroids(image_array, centroids):
    K = np.size(centroids, 0)
    indexes = np.zeros((np.size(image_array, 0), 1))
    arr = np.empty((np.size(image_array, 0), 1))
    for i in range(0, K):
        each_cent = centroids[i]
        temp = np.ones((np.size(image_array, 0), 1)) * each_cent
        dist_square = np.power(np.subtract(image_array, temp), 2)
        a = np.sum(dist_square, axis=1)
        a = np.asarray(a)
        a.resize((np.size(image_array, 0), 1))
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr, 0, axis=1)
    indexes = np.argmin(arr, axis=1)
    return indexes

for file in filename:
	# Image to 2D array
	image = io.imread(file)  # image is saved as rows * columns * 3 array
	rows = image.shape[0]
	columns = image.shape[1]
	image = image.astype(np.float64)
	image = image / 255
	image_array = image.reshape(image.shape[0] * image.shape[1], 3)

	# K Means start here
	for each_K in K:
		centroids = random.sample(list(image_array), each_K)  # randomly picking each_K number of centroids from the array
		length = np.size(image_array, 0)
		width = np.size(image_array, 1)
		indexes = np.zeros((length, 1))  # empty array of length of image_array
		for i in range(1, max_iterations):
			indexes = closest_centroids(image_array, centroids)
			new_centroids = np.zeros((each_K, width))
			for j in range(0, each_K):
				centroid_index = indexes == j
				centroid_index = centroid_index.astype(int)
				total_number = sum(indexes == j)
				centroid_index.resize((np.size(image_array, 0), 1))
				total_matrix = nm.repmat(centroid_index, 1, width)
				centroid_index = np.transpose(centroid_index)
				total = np.multiply(image_array, total_matrix)
				new_centroids[j] = (1 / total_number) * np.sum(total, axis=0)
			centroids = new_centroids

		indexes = closest_centroids(image_array, centroids)
		new_image_array = centroids[indexes]
		new_image_array = np.reshape(new_image_array, (rows, columns, 3))
		print('Processed {} for K: {}'.format(file, each_K))

		# Array to image file
		scipy.misc.imsave('{}_{}.jpg'.format(file,each_K), new_image_array)