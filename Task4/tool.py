#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: 罗梓颖


import numpy as np

def get_batch(data,label,batch_size):
	i = 0
	while (i + batch_size) <= len(data):
		yield data[i:i+batch_size],label[i:i+batch_size]
		i += batch_size
	yield data[i:],label[i:]
	return None

def array_map(array):
	name = np.unique(array)
	y_num = [i for i in map(lambda x:np.where(x == name)[0][0],array)]
	return y_num,name