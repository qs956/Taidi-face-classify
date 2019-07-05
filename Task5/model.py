#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: 罗梓颖

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

pic_size = (256,256)
name_map = np.load(r'../Task4/model/name_map.npy')


class FaceDetector:
	def __init__(self, model_path='../Task2/model.pb', gpu_memory_fraction=0.75, visible_device_list='0'):
		"""
		Arguments:
			model_path: a string, path to a pb file.
			gpu_memory_fraction: a float number.
			visible_device_list: a string.
		"""
		with tf.gfile.GFile(model_path, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())

		graph = tf.Graph()
		with graph.as_default():
			tf.import_graph_def(graph_def, name='import')

		self.input_image = graph.get_tensor_by_name('import/image_tensor:0')
		self.output_ops = [
			graph.get_tensor_by_name('import/boxes:0'),
			graph.get_tensor_by_name('import/scores:0'),
			graph.get_tensor_by_name('import/num_boxes:0'),
		]
		config_proto = tf.ConfigProto(log_device_placement=False)
		self.sess = tf.Session(graph=graph, config=config_proto)

	def __call__(self, image, score_threshold = 0.75):
		"""Detect faces.

		Arguments:
			image: a numpy uint8 array with shape [height, width, 3],
				that represents a RGB image.
			score_threshold: a float number.
		Returns:
			boxes: a float numpy array of shape [num_faces, 4].
			scores: a float numpy array of shape [num_faces].

		Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
		"""
		h, w, _ = image.shape
		image = np.expand_dims(image, 0)

		boxes, scores, num_boxes = self.sess.run(
			self.output_ops, feed_dict={self.input_image: image}
		)
		num_boxes = num_boxes[0]
		boxes = boxes[0][:num_boxes]
		scores = scores[0][:num_boxes]

		to_keep = scores > score_threshold
		boxes = boxes[to_keep]
		scores = scores[to_keep]

		scaler = np.array([h, w, h, w], dtype='float32')
		boxes = boxes * scaler

		return boxes, scores

# class recon_pre:
# 	def __init__(self,output_size = (256,256)):
# 		self.output_size = output_size
# 	def __call__(self,img):
# 		return cv2.resize(img,output_size)

class ResNet_recon:
	def __init__(self,mu_std_path = r'../Task4/model/mu_std.npy',graph_path = r'../Task4/model//model.meta',ckpt_path = r'../Task4/model/'):
		temp = np.load(mu_std_path)
		self.mu,self.sigma = temp[0],temp[1]

		self.G = tf.Graph()
		with self.G.as_default():
			self.sess = tf.Session(graph = self.G)
			self.saver = tf.train.import_meta_graph(graph_path)
			self.inputs = self.sess.graph.get_tensor_by_name('x_input:0')
			self.y_pred = self.sess.graph.get_tensor_by_name('output/y_pred:0')
			self.y_proba = self.sess.graph.get_tensor_by_name('output/y_proba:0')
			self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_path))

	def __call__(self,inputs):
		# inputs -= self.mu.astype(np.float32)
		# inputs /= self.sigma.astype(np.float32)
		y_pred,y_proba = self.sess.run([self.y_pred,self.y_proba],feed_dict = {self.inputs:inputs})
		return y_pred,y_proba

def main(img,face_detect,face_recon,name):
	#人脸检测
	boxes,scores = face_detect(img)
	#灰度处理
	if (len(scores) != 0):
		ymin,xmin,ymax,xmax = boxes[np.argmax(scores)]
		pic_crop = Image.fromarray(img).crop([xmin,ymin,xmax,ymax])
		pic_crop = np.asarray(pic_crop.convert('L'))
		cv2.imshow("gray", pic_crop)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows() 
		# pic_crop = img[int(ymin):int(ymax),int(xmin):int(xmax)]
		# pic_crop = cv2.cvtColor(pic_crop,cv2.COLOR_BGR2GRAY)
	else:
		return img
	cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255), 4)
	#数据预处理
	pic_crop = cv2.resize(pic_crop,pic_size)
	# pic_crop = pic_crop.astype(np.float32)
	#人脸识别
	y_pred,y_proba = face_recon(pic_crop.reshape((1,pic_size[0]*pic_size[1])))
	print('pred:',name[y_pred][0],'proba:',y_proba[0])
	text = str(name[y_pred][0]) + str(y_proba[0])
	cv2.putText(img, text, (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
	return img


if __name__ == '__main__':
	print('正在加载模型,请稍后......')
	face_detect = FaceDetector()
	face_recon = ResNet_recon()
	name = name_map[0]
	print('模型加载完成!')
	cap = cv2.VideoCapture(0)
	while(1):
		ret, frame = cap.read()
		img = main(frame,face_detect,face_recon,name)
		cv2.imshow("按q退出", img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()#释放摄像头
	cv2.destroyAllWindows() 