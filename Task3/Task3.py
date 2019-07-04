#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: 罗梓颖

import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

	save_path = r'result/'
	random_state = 0
	pic_size = (256,256)
	test_size = 0.2

	path = r'../Task2/faceimageGray/'
	person = os.listdir(path)


	#read photo
	train_data = []
	train_targe = []
	for i in person:
		person_path = path + i + r'/'
		file_all = os.listdir(person_path)
		for j in file_all:
			pic = plt.imread(person_path + j)
			train_data.append(pic)
			train_targe.append(i)




	train_data = [i for i in map(lambda x:np.reshape(cv2.resize(x,pic_size),-1),train_data)]




	x_train,x_test,y_train,y_test = train_test_split(train_data,train_targe,test_size = test_size, random_state = random_state)


	#Normalize
	# x_train = np.array(x_train,dtype = np.float32)
	# x_test = np.array(x_test,dtype = np.float32)

	# mu = np.mean(x_train,axis = 0)
	# sigma = np.std(x_train,axis = 0)


	# x_train -= mu
	# x_train /= sigma

	# x_test -= mu
	# x_test /= sigma


	np.save(save_path + r'/train_data.npy',x_train)
	np.save(save_path + r'/train_label.npy',y_train)
	np.save(save_path + r'/test_data.npy',x_test)
	np.save(save_path + r'/test_label.npy',y_test)
