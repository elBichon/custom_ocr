import utils
import numpy as np
import statistics 
import math



def test_get_size():
	assert utils.get_size([8, 14, 13, 8, 42, 7, 6, 9, 20, 41, 40, 40, 43, 44, 45, 43, 43, 43]) == 29

def test_compute_dist_between_characters():
	dist = utils.compute_dist_between_characters([1,2,3,4,5])
	assert dist == [1,1,1,1]

def test_compute_std_dist():
	assert utils.compute_std_dist([1,1,1,1]) == int(0)

def test_correct_sentence():
	assert utils.correct_sentence('tha1a2t') == 'that'

