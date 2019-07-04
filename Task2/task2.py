#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: 罗梓颖

import os
from PIL import Image
import numpy as np
import faceboxes_detector as fa
from tqdm import tqdm

save_path = r'faceimageGray/'


if __name__ == '__main__':
	if not(os.path.exists(save_path)):
		os.mkdir(save_path)

	dector = fa.FaceDetector()

	path = r'../Task1/faceimages/'
	person = os.listdir(path)

	for i in person:
		person_path = path + i + '/'
		file_all = os.listdir(person_path)

		if not(os.path.exists(save_path + i + '/')):
			os.mkdir(save_path + i + '/')

		for j in tqdm(file_all):
			with Image.open(person_path + j) as pic:
				pic_array = np.asarray(pic)
				boxes,scores = dector(pic_array)
				if (len(scores) != 0):
					ymin,xmin,ymax,xmax = boxes[np.argmax(scores)]
					pic_crop = pic.crop([xmin,ymin,xmax,ymax])
					pic_crop = pic_crop.convert('L')
					pic_crop.save(save_path + i + r'/' + j + r'.jpg')
	print('转换完成!')
	os.system(r'pause')
