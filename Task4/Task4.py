#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: 罗梓颖


import numpy as np
import tensorflow as tf
import resnet_v2 as res
from tool import *

epoch = 3
batch_size = 16
log_path = r'log/'
model_path = r'./model/model'

train_data = np.load(r'../Task3/result/train_data.npy')
train_label = np.load(r'../Task3/result/train_label.npy')

# train_data = train_data[0:500]
# train_label = train_label[0:500]

#Normalize
mu = np.mean(train_data,axis = 0)
std = np.std(train_data,axis = 0)
np.save(r'model/mu_std.npy',(mu,std))


y_num,name = array_map(train_label)
np.save(r'model/name_map.npy',(name,[i for i in range(len(name))]))
print('Pre-Process parameter has saved')


G = tf.Graph()
with G.as_default():
    x = tf.placeholder(tf.uint8,shape = [None,256*256],name = 'x_input')
    y = tf.placeholder(tf.uint8,shape = [None],name = 'y_input')

    def pre_process(x,y):
        x = tf.reshape(x,[-1,256,256,1])
        x_norm = tf.cast(x,tf.float32)
        y_norm = tf.one_hot(y,depth = 10)
        y_norm = tf.reshape(y_norm,[-1,10])
        return x_norm,y_norm

    x_norm,y_norm = pre_process(x,y)
    resnet = res.resnet_v2_50(x_norm,num_classes = 10)
    loss = tf.losses.softmax_cross_entropy(logits = resnet[0],onehot_labels = y_norm)
    with tf.name_scope("output"):
        output_layer = tf.nn.softmax(resnet[0],axis = 1,name = 'output_layer')
        y_pred = tf.argmax(output_layer,axis = 1, name='y_pred')
        y_proba = tf.div(tf.reduce_max(output_layer,axis = 1),tf.reduce_sum(output_layer,axis = 1),name = 'y_proba')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y,tf.int64),y_pred),tf.float32),name='accuracy')
    tf.summary.scalar('Total_loss',loss)
    tf.summary.scalar('Accuracy',accuracy)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_path, G)
    train_op = tf.train.AdamOptimizer().minimize(loss)


print('Start train....')
with tf.Session(graph = G) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(epoch):
        data_gen = get_batch(train_data,y_num,batch_size)
        for j_num,j in enumerate(data_gen):
            data,label = j[0],j[1]
            _,show_loss = sess.run([train_op,loss],feed_dict = {x:data,y:label})
            if (j_num%10 == 0):
                summary,acc = sess.run([merged,accuracy],feed_dict = {x:data,y:label})
                writer.add_summary(summary, i*np.ceil(train_data.shape[0]/batch_size)+j_num)
                print(f'epoch:{i+1}/{epoch},step:{j_num},Batch loss:{show_loss},Batch Accuracy:{acc}')
    # saver = tf.train.Saver()
    saver.save(sess,model_path)

print('The train has finished')