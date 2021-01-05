import cv2  
import argparse
from math import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import os
import statistics 


def get_img(image):
	try:
		img = cv2.imread(image)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
		return img
	except:
		return False

def extract_images(img):
# Performing OTSU threshold 
	ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
	# Kernel size increases or decreases the area  
	# of the rectangle to be detected. 
	# A smaller value like (10, 10) will detect  
	# each word instead of a sentence. 
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) 
	# Appplying dilation on the threshold image 
	dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
	# Finding contours 
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
	# Creating a copy of image 
	return contours

def crop_img(contours,img):
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt) 
		# Drawing a rectangle on copied image 
		rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
		# Cropping the text block for giving input to OCR 
		cropped = img[y:y + h, x:x + w]
		return cropped

def get_shapes_array(contours,img):
	shapes_array = []
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt) 
		rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
		cropped = img[y:y + h, x:x + w]
		shapes_array.append(cropped.shape[0])
	return shapes_array

def get_size(shapes_array):
    return ceil(sum(shapes_array)/len(shapes_array))

def crop_to_avg_size(img,size,dsize,contours):
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt) 
		# Drawing a rectangle on copied image 
		rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
		# Cropping the text block for giving input to OCR 
		cropped = img[y:y + h, x:x + w]
		if cropped.shape[0] >= size:
			cropped = cv2.resize(cropped, dsize)
			cv2.imwrite('to_translate/'+str(x)+'.png',cropped)
			cv2.destroyAllWindows()
				
def vectorize_img(img):
	arr = np.array(img)
	img_vector = arr.ravel()
	return list(img_vector)

def compute_cosine_similarity(img,img_alphabet):
	dist = cosine_similarity([img], [img_alphabet])
	return dist

def sort_images(images_list):
	i = 0
	while i < len(images_list):
		images_list[i] = int(images_list[i].replace('.png',''))
		i += 1
	images_list = sorted(images_list)
	return images_list

def convert_img_to_string(images_list):
	i = 0
	while i < len(images_list):
		images_list[i] = str(images_list[i])+'.png'
		i += 1
	return images_list

def get_letter(letter,metric):
	return letter[int(metric.index(max(metric)))]

def compute_dist_between_characters(images_coordinates):
	i = 1
	dist = []
	while i < len(images_coordinates):
		dist.append(images_coordinates[i]-images_coordinates[i-1])
		i += 1
	return dist

def compute_avg_dist(dist):
	return math.ceil(sum(dist)/len(dist))

def compute_std_dist(dist):
	return statistics.stdev(dist)

def recreate_sentences(words,dist_between_characters,images_list):
	i = 1
	sentence = ''
	while i <= len(words):
		if i != len(words) and images_list[i]-images_list[i-1] < dist_between_characters:
			sentence = sentence + words[i-1]
		elif i != len(words) and images_list[i]-images_list[i-1] > dist_between_characters:
			sentence = sentence + words[i-1] + ' '
		else:
			sentence = sentence + words[i-1]
		i += 1
	return sentence 

def correct_sentence(sentence):
	return sentence.replace('a1a2','a')

