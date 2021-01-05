import cv2  
import argparse
from math import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
import os
import utils
import statistics 

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='running')
	parser.add_argument('image', type=str, help='choosing a picture')
	args = parser.parse_args()
	image  = args.image

	DSIZE = (41,22)

	img = utils.get_img(image)
	contours = utils.extract_images(img)
	cropped = utils.crop_img(contours,img)
	shapes_array = utils.get_shapes_array(contours,img)
	size = utils.get_size(shapes_array)
	utils.crop_to_avg_size(img,size,DSIZE,contours)
	img_path = 'to_translate/'
	images = os.listdir(img_path)
	images_coordinates = utils.sort_images(images)
	dist_list = utils.compute_dist_between_characters(images_coordinates)
	avg_dist = utils.compute_avg_dist(dist_list)
	std_dist = utils.compute_std_dist(dist_list)
	dist_between_characters = avg_dist+std_dist
	images = utils.convert_img_to_string(images_coordinates)

	words = []
	for imgs in images:
		img = cv2.imread('to_translate/'+imgs)
		img_to_translate = utils.vectorize_img(img)
		letter = []
		metric = []
		img_target = os.listdir('alphabet/')
		for img in img_target:
			img_alphabet = cv2.imread('alphabet/'+str(img))
			img_alphabet = utils.vectorize_img(img_alphabet)
			dist = utils.compute_cosine_similarity(img_to_translate,img_alphabet)
			letter.append(str(img).replace('.png',''))
			metric.append(dist[0][0])
		words.append(utils.get_letter(letter,metric))
	images = os.listdir(img_path)
	images_coordinates = utils.sort_images(images)
	recreated_sentence = utils.recreate_sentences(words,dist_between_characters,images_coordinates)
	print(utils.correct_sentence(recreated_sentence))






















   



