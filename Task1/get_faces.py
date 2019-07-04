#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: 罗梓颖

import os
import msvcrt
import cv2


import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
	print(r'---------------------拍照片---------------------')
	print(r'Author: 罗梓颖')
	try:
		cap = cv2.VideoCapture(0)
	except:
		print('摄像头调用失败!')
		while True:
			cap_num = int(input('请输入摄像头编号：'))
			try:
				cap = cv2.VideoCapture(cap_num)
				break
			except:
				print('摄像头调用失败!')
	pic_num = int(input('请输入要拍摄的数量：'))
	pic_pre = input('请输入人名：')
	print('请摆好POSS,按下Enter键开始拍照!')
	while True:
		key = ord(msvcrt.getch())
		if (key == 13):
			break
		else:
			pass
	for i in tqdm(range(pic_num)):
		sucess,img = cap.read()
		if not (sucess):
			print('图片获取失败!')
			break
		else:
			cv2.imwrite(r'faceimages/' + pic_pre + r'/' + str(i) + r'.jpg',img)
	print('图片获取成功！')
	os.system(r'pause')