#!/usr/bin/python

import cv2, os
import sys
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
import skimage

np.set_printoptions(threshold=np.nan)


def main():
	path = './Lindex101/'
	paths = [os.path.join(path, f) for f in os.listdir(path)]
	paths.sort()
	paths = [paths[i:i+4] for i in range(0,len(paths),4)]

	subjects = []
	for path_list in paths:
			new_subj = Subject(path_list)
			subjects.append(new_subj)


	for subject in subjects:
		subject.get_features()

	for subject1 in subjects:
		for subject2 in subjects:
			for sample1 in subject1.sample_list:
				for sample2 in subject2.sample_list:
					matching (sample1, sample2)


class Sample:
	def __init__(self, label, image, orientations, cores, deltas, fingerprint_type, thinned, minutias):
		self.label = label
		self.image = image
		self.orientations = orientations
		self.cores = cores
		self.deltas = deltas
		self.fingerprint_type = fingerprint_type
		self.thinned = thinned
		self.minutias = minutias


class Subject:
	def __init__(self, path_list):
		self.path_list = path_list
		self.sample_list = []

	def get_features(self):
		for path in self.path_list:
			image = np.empty((300,300), np.uint8)
			image.data[:] = open(path).read()

			label = path[12:-4]
			print 'reading {}'.format(path)
			show_image (image, label)

			enhanced = enhance_image(image)
			show_image (enhanced, label + '_enhanced')

			roi = get_roi (enhanced, w_size=10)
			show_roi(enhanced, roi, label + '_roi')

			orientations, alpha_x, alpha_y = compute_orientation(enhanced, w=10)
			show_orientations(enhanced, orientations, roi, label + '_orientations', w=10)

			deltas, cores = singular_point_detection (alpha_x, alpha_y, w_size=10, image=enhanced, roi=roi)
			fingerprint_type = get_fingerprint_type(deltas, cores)
			show_singular_points(deltas, cores, fingerprint_type, enhanced, 10, label + '_singular')

			binarized = binarization(enhanced, 10, roi)
			show_image(binarized, label + '_binarized')

			smoothed = smoothing (binarized)
			show_image(smoothed, label + '_smoothed')

			thinned = thinning (smoothed)
			show_image(thinned, label + '_thinned')

			ridges = minutiae_detection(thinned)
			show_ridges (ridges, thinned, label + 'ridges')
			
			extracted_minutia = minutiae_extraction(ridges, thinned, roi)

			sample = Sample(label, enhanced, orientations, cores, deltas, fingerprint_type, thinned, extracted_minutia)
			self.sample_list.append(sample)

#matching (enhanced, cores, fingerprint_type, orientations)
def matching (sample1, sample2):
	print ('matching score ({},{}):').format(sample1.label, sample2.label),
	sys.stdout.flush()


	#translation
	center1 = (sample1.cores[0][0], sample1.cores[0][1])
	center2 = (sample2.cores[0][0], sample2.cores[0][1])
	
	endings1 = translate_matrix(sample1.minutias[1], center1, center2)
	edges1 = translate_matrix(sample1.minutias[2], center1, center2)
	bifurcations1 = translate_matrix(sample1.minutias[3], center1, center2)
	crossings1 = translate_matrix(sample1.minutias[4], center1, center2)


	#rotation
	angle_offset = np.degrees(sample1.orientations[center1[1],center1[0]] - sample2.orientations[center2[1],center2[0]])
	print 'angle_offset:{}'.format(angle_offset)
	endings1 = rotate_matrix (endings1, center1, angle_offset)
	edges1 = rotate_matrix (edges1, center1, angle_offset)
	bifurcations1 = rotate_matrix (bifurcations1, center1, angle_offset)
	crossings1 = rotate_matrix (crossings1, center1, angle_offset)


	endings2 = sample2.minutias[1]
	edges2 = sample2.minutias[2]
	bifurcations2 = sample2.minutias[3]
	crossings2 = sample2.minutias[4]

	
	#matching
	final_score = 0
	final_count = 0

	score, M = match_matrixes (endings1, endings2, radius=8)
	final_score += score
	final_count += M
		
	score, M = match_matrixes (edges1, edges2, radius=8)
	final_score += score
	final_count += M

	score, M = match_matrixes (bifurcations1, bifurcations2, radius=12)
	final_score += score
	final_count += M

	score, M = match_matrixes (crossings1, crossings2, radius=12)
	final_score += score
	final_count += M

	print final_score



def match_matrixes (matrix1, matrix2, radius):
	score = 0
	M = 0
	points1 = np.argwhere(matrix1 > 0)
	points2 = np.argwhere(matrix2 > 0)

	for point1 in points1:
		for point2 in points2:
			dif1 = point1[0] - point2[0]
			dif2 = (point1[1] - point2[1])
			distance = math.sqrt((dif1*dif1) + (dif2*dif2))
			if distance <= radius:
				score += (1 - distance/radius)
				M += 1

	return score, M

def rotate_matrix (matrix, center, angle):
	M = cv2.getRotationMatrix2D((center[1], center[0]), angle, 1)
	rotated = cv2.warpAffine(matrix, M, (300,300), flags=cv2.INTER_NEAREST)

	return rotated



def translate_matrix(matrix, center1, center2):
	offset_x = center2[0] - center1[0]
	offset_y = center2[1] - center1[1]

	M = np.float32([[1,0,offset_y],[0,1,offset_x]]) #transformation matrix for translation
	translated = cv2.warpAffine(matrix, M, (300,300))
	
	return translated


def show_ridges (ridges, thinned, label):
	show_points(ridges[0], thinned, label + "_0")
	show_points(ridges[1], thinned, label + "_1")
	show_points(ridges[2], thinned, label + "_2")
	show_points(ridges[3], thinned, label + "_3")
	show_points(ridges[4], thinned, label + "_4")

def minutiae_extraction(ridges, image, roi):
	isolated, ending, edge, bifurcation, crossing = ridges
	
	roi = cv2.bitwise_not(roi)

	ending_conv = cv2.filter2D(ending, -1, np.ones((8,8), dtype=np.uint8))
	bifurcation_conv = cv2.filter2D(bifurcation, -1, np.ones((8,8), dtype=np.uint8))
	roi_conv = cv2.filter2D(roi, -1, np.ones((10,10), dtype=np.uint8))

	#Two endings are too close (within 8 pixels)
	ending[ending_conv >= 2] = 0

	#An ending and a bifurcation are too close
	bifurcation[(ending_conv) > 0 & (bifurcation_conv > 0)] = 0
	ending[(ending_conv > 0) & (bifurcation_conv > 0)] = 0

	#Two bifurcations are too close 
	bifurcation[bifurcation_conv >= 2] = 0

	#Minutiae are near the margins
	ending[roi_conv > 0] = 0
	edge[roi_conv > 0] = 0
	bifurcation[roi_conv > 0] = 0
	crossing[roi_conv > 0] = 0

	return [isolated, ending, edge, bifurcation, crossing]


def minutiae_detection(image):
	image = cv2.bitwise_not(image) / 255
	kernel = np.array(([1,1,1],[1,0,1],[1,1,1]), np.uint8)
	connected_neighbours = cv2.filter2D(image, -1, kernel)

	isolated	= np.uint8(connected_neighbours == 0) * 1
	ending 		= np.uint8(connected_neighbours == 1) * 1
	edge		= np.uint8(connected_neighbours == 2) * 1
	bifurcation = np.uint8(connected_neighbours == 3) * 1
	crossing	= np.uint8(connected_neighbours == 4) * 1

	isolated    = cv2.bitwise_and(image, isolated)
	ending 	    = cv2.bitwise_and(image, ending)
	edge	    = cv2.bitwise_and(image, edge)
	bifurcation = cv2.bitwise_and(image, bifurcation)
	crossing    = cv2.bitwise_and(image, crossing)

	return [isolated, ending, edge, bifurcation, crossing]

def show_points(points_matrix, image, path):
	cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	points = np.argwhere(points_matrix > 0)

	for point in points:
		cv2.circle(cimg, (point[1],point[0]), 1, color=(0,0,255), thickness=2)

	cv2.putText(cimg,path,(00,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,100,255),2,True)
	cv2.imshow('norm', cimg)
	cv2.waitKey(0)


def thinning (image):
	skeletonized = skeletonize(cv2.bitwise_not(image) / 255)
	skeletonized = np.uint8(skeletonized)

	return cv2.bitwise_not(skeletonized * 255)

def smoothing (image):
	smoothed = image / 255

	#first smooth filter
	kernel = np.ones((5,5), np.uint8)
	white_pixels = cv2.filter2D(smoothed, -1, kernel)
	black_pixels = 25 - white_pixels
	
	smoothed[white_pixels >= 18] = 1
	smoothed[black_pixels >= 18] = 0


	#last smooth filter
	kernel = np.ones((3,3), np.uint8)
	white_pixels = cv2.filter2D(smoothed, -1, kernel)
	black_pixels = 9 - white_pixels

	smoothed[white_pixels >= 5] = 1
	smoothed[black_pixels >= 5] = 0

	return smoothed * 255

def binarization(image, w, roi):
	hist, bins = np.histogram(image.ravel(), 256, [0,256])
	
	#calculating P25 and P50
	sum = 0
	p25 = 0
	p50 = 0
	max = image.shape[0] * image.shape[1]
	for i in xrange(0,256):
		sum += hist[i]
		if sum >= max * 0.25 and not p25:
			p25 = i
		if sum >= max * 0.50 and not p50:
			p50 = i
			break


	mean_blocks_matrix = np.zeros(image.shape)
	for i in np.arange(0,300,w):
		for j in np.arange(0,300,w):
			submatrix = image[i:i+w,j:j+w]
			mean_blocks_matrix[i:i+w,j:j+w] = np.mean(submatrix)

	kernel = np.array(([1,1,1],[1,0,1],[1,1,1]), np.float32) / 8
	mean_around_pixel = cv2.filter2D(image, -1, kernel)

	binarized = np.ones(image.shape, np.uint8) * 255
	for i in xrange(0,300):
		for j in xrange(0,300):
			if roi[i,j]:
				if image[i,j] < p25:
					binarized[i,j] = 0
				elif image[i,j] > p50:
					binarized[i,j] = 255
				else:
					binarized[i,j] = 255 if mean_around_pixel[i,j] >= mean_blocks_matrix[i,j] else 0

	return binarized



def diff_ang(a,b):
	diff = a - b

	if (diff >= -math.pi/2 and diff <= math.pi/2):
		return diff
	elif diff > math.pi/2:
		return diff - math.pi
	elif diff < -math.pi/2:
		return diff + math.pi

def singular_point_detection(alpha_x, alpha_y, w_size, image, roi):
	#block reduce from [300,300] to [30,30] (w = 10)
	alpha_x = block_reduce(alpha_x, w_size)
	alpha_y = block_reduce(alpha_y, w_size)

	#erode the roi a little
	kernel = np.ones((5,5),np.uint8)
	roi = cv2.erode(roi, kernel, iterations = 4)

	#smooth directions
	kernel = np.array(([1,1,1],[1,2,1],[1,1,1]))
	a = cv2.filter2D(alpha_x, -1, kernel)
	b = cv2.filter2D(alpha_y, -1, kernel)


	theta_avg = np.arctan2(b, a) * 0.5 + (math.pi / 2)

	poincare_index = np.zeros(theta_avg.shape)
	for i in np.arange(1,theta_avg.shape[0]-1):
		for j in np.arange(1,theta_avg.shape[1]-1):
			if roi[i*w_size,j*w_size]:
				poincare_index[i,j]  = diff_ang(theta_avg[i-1,j-1], theta_avg[i-1,j])
				poincare_index[i,j] += diff_ang(theta_avg[i-1,j], theta_avg[i-1,j+1])
				poincare_index[i,j] += diff_ang(theta_avg[i-1,j+1], theta_avg[i,j+1])
				poincare_index[i,j] += diff_ang(theta_avg[i,j+1], theta_avg[i+1,j+1])
				poincare_index[i,j] += diff_ang(theta_avg[i+1,j+1], theta_avg[i+1,j])
				poincare_index[i,j] += diff_ang(theta_avg[i+1,j], theta_avg[i+1,j-1])
				poincare_index[i,j] += diff_ang(theta_avg[i+1,j-1], theta_avg[i,j-1])
				poincare_index[i,j] += diff_ang(theta_avg[i,j-1], theta_avg[i-1,j-1])


	cores = (poincare_index < np.radians(-135)) & (poincare_index > np.radians(-225)) * 1
	cores = reduce_blobs(cores)

	deltas = (poincare_index > np.radians(135)) & (poincare_index < np.radians(225)) * 1
	deltas = reduce_blobs(deltas)

	return deltas*w_size, cores*w_size

def get_fingerprint_type(deltas, cores):
	number_of_cores = cores.shape[0]
	number_of_deltas = deltas.shape[0]

	if (number_of_deltas >= 0) and (number_of_cores == 0):
		return 'arch'
	if number_of_cores >= 2:
		return 'whorl'
	if (number_of_deltas == 1) and (number_of_cores == 1):
		if deltas[0][1] > cores[0][1]:
			return 'left_loop'
		else:
			return 'right_loop'
	return 'undefined'


def show_singular_points(deltas, cores, fingerprint_type, image, w, path):
	cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	for point in deltas:
		cv2.circle(cimg, (point[1],point[0]), 1, color=(255,0,0), thickness=2)
		#print 'delta[{},{}]'.format(point[0],point[1])

	for point in cores:
		cv2.circle(cimg, (point[1],point[0]), 1, color=(0,0,255), thickness=2)		
		#print 'core[{},{}]'.format(point[0],point[1])

	cv2.putText(cimg,fingerprint_type,(20,290), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,100,25),2,True)
	cv2.putText(cimg,path,(00,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,100,255),2,True)
	cv2.imshow('norm', cimg)
	cv2.waitKey(0)


def reduce_blobs(matrix):
	blob_list = []
	matrix = np.uint8(matrix)

	while np.argwhere(matrix).shape[0] > 0:
		points = np.argwhere(matrix)
		blob_list.append(points[0])
		mask = np.zeros((32, 32), np.uint8)
		cv2.floodFill(matrix, mask, (points[0][1],points[0][0]), 0)

	return np.asarray(blob_list)



def get_end_points(point, angle, length):
     x, y = point

     endy = int(y + length * math.sin(angle))
     endx = int(x + length * math.cos(angle))

     return endx, endy


def block_gradient_direction(alpha_x, alpha_y):
	orientations = np.arctan2(alpha_y, alpha_x)

	return np.divide(orientations, 2) + (math.pi / 2)

def block_reduce (matrix, w):
	return matrix[0:matrix.shape[0]:w, 0:matrix.shape[1]:w]


def compute_orientation(image, w):
	#image = cv2.medianBlur(image, 5)

	Gx = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
	Gy = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

	alpha_x = np.square(Gx) - np.square(Gy)
	alpha_y = 2 * Gx * Gy

	alpha_x = cv2.filter2D(alpha_x, -1, np.ones((w,w)), anchor=(0,0)) / (w*w)
	alpha_y = cv2.filter2D(alpha_y, -1, np.ones((w,w)), anchor=(0,0)) / (w*w)

	orientations = block_gradient_direction(alpha_x, alpha_y)

	return orientations, alpha_x, alpha_y

def show_orientations(image, orientations, roi, path, w):
	cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	for i in np.arange(0,orientations.shape[0],w):
		for j in np.arange(0,orientations.shape[1],w):
			if roi[i,j]:
				cv2.line(cimg, pt1=(j,i), pt2=get_end_points((j,i), orientations[i,j], 5), color=(0,0,255), thickness=2)

	cv2.putText(cimg,path,(00,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,100,255),2,True)
	cv2.imshow('norm', cimg)
	cv2.waitKey(0)


def enhance_image(image):
	mean = np.mean(image)
	std_variance = np.std(image)

	enhanced = 150. + 95. * ((image - mean) / float(std_variance))
	
	for i in xrange(0,300):
		for j in xrange(0,300):
			if enhanced[i,j] < 0:
				enhanced[i,j] = 0
			elif enhanced[i,j] > 255:
				enhanced[i,j] = 255

	enhanced = np.uint8(enhanced)

	return enhanced

def get_roi(image, w_size):
	mean_array = []
	std_array = []

	roi = np.zeros(image.shape, dtype='uint8')

	for i in np.arange(0,300,w_size):
		for j in np.arange(0,300,w_size):
			submatrix = image[i:i+w_size,j:j+w_size]

			mean_array.append(np.mean(submatrix))
			std_array.append(np.std(submatrix))

	mean_array = np.asarray(mean_array)
	std_array = np.asarray(std_array)

	mean_max = np.amax(mean_array)
	std_max =  np.amax(std_array)

	for i in np.arange(0,300,w_size):
		for j in np.arange(0,300,w_size):
			submatrix = image[i:i+w_size,j:j+w_size]

			mean = np.mean(submatrix)
			std = np.std(submatrix)

			dist_ratio = 1 / np.linalg.norm(np.array([151,151])-np.array([i,j]))
			v = 0.5 * (1 - mean/mean_max) + 0.5 * std/std_max + 10*dist_ratio
			
			if v > 0.3:
				roi[i:i+w_size, j:j+w_size] = 255


	im_floodfill = roi.copy()
	mask = np.zeros((302, 302), np.uint8)
	cv2.floodFill(im_floodfill, mask, (0,0), 255)
	im_floodfill = cv2.bitwise_not(im_floodfill)
	roi = cv2.bitwise_or(roi, im_floodfill)

	kernel = np.ones((5,5),np.uint8)
	roi = cv2.erode(roi, kernel, iterations = 1)

	return roi

def show_roi (image, roi, path):
	img = cv2.bitwise_and(image, roi)
	cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.putText(cimg,path,(00,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,100,255),2,True)
	cv2.imshow('norm', cimg)
	cv2.waitKey(0)

def show_image (image, path):
	cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	cv2.putText(cimg,path,(00,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,100,255),2,True)
	cv2.imshow('norm', cimg)
	cv2.waitKey(0)

if __name__ == "__main__":
	main()