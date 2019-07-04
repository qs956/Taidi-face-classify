#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: 罗梓颖


import numpy as np
import tensorflow as tf
import resnet_v2 as res

epoch = 5
batch_size = 16
log_path = r'log/'
model_path = r'./model/model.ckpt'

train_data = np.load(r'../Task3/result/train_data.npy')
train_label = np.load(r'../Task3/result/train_label.npy')

#Normalize
mu = np.mean(train_data,axis = 0)
std = np.std(train_data,axis = 0)
np.save(r'model/mu_std.npy',(mu,std))


name = np.unique(train_label)
y_num = [i for i in map(lambda x:np.where(x == name)[0][0],train_label)]
np.save(r'model/name_map.npy',(name,[i for i in range(len(name))]))
print('Pre-Process parameter has saved')

def get_batch(data,label,batch_size):
    i = 0
    while (i + batch_size) <= len(data):
        yield data[i:i+batch_size],label[i:i+batch_size]
        i += batch_size
    yield data[i:],label[i:]
    return None


G = tf.Graph()
with G.as_default():
    def get_input():
        x = tf.placeholder(tf.uint8,shape = [None,256*256])
        y = tf.placeholder(tf.uint8,shape = [None])
        return x,y

    def pre_process(x,y):
        x = tf.reshape(x,[-1,256,256,1])
        x_norm = tf.cast(x,tf.float32)
        y_norm = tf.one_hot(y,depth = 10)
        y_norm = tf.reshape(y_norm,[-1,10])
        return x_norm,y_norm
    x,y = get_input()
    x_norm,y_norm = pre_process(x,y)
    resnet = res.resnet_v2_50(x_norm,num_classes = 10)
    loss = tf.losses.softmax_cross_entropy(logits = resnet[0],onehot_labels = y_norm)
    tf.summary.scalar('Total_loss',loss)
    y_pred = tf.argmax(resnet[0])
    train_op = tf.train.AdamOptimizer().minimize(loss)


writer = tf.summary.FileWriter(log_path, G)

print('Start train....')
with tf.Session(graph = G) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        data_gen = get_batch(train_data,y_num,batch_size)
        for j_num,j in enumerate(data_gen):
            data,label = j[0],j[1]
            _,show_loss = sess.run([train_op,loss],feed_dict = {x:data,y:label})
            if (j_num%10 == 0):
                print(f'epoch:{i},step:{j_num},loss:{show_loss}')
    saver = tf.train.Saver()
    saver.save(sess,model_path)

print('The train has finished')