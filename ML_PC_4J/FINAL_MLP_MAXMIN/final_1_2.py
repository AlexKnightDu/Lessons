#import pudb
#pu.db

import tensorflow as tf
import numpy as np
import scipy.io as scio
#import pandas as pd
import math as ma
from tensorflow.contrib import layers
import time



def next_batch(data, batch_size, i):
    s = len(data)
    if (i + 1) * batch_size > s:
        return data[i*batch_size:-1]
    else:
        return data[i*batch_size:(i+1) * batch_size]

def onehot(labels, units):
    l = len(labels)
    onehot_labels = np.zeros([l,units])
    for i in range(0,l):
        onehot_labels[i][labels[i][0]] = 1
    return onehot_labels

def normalize(data, base):
    min_datum = []
    max_datum = []
    base = np.array(base)
    for i in range(len(base[0])):
        min_datum += [min(base[:,i])]
        max_datum += [max(base[:,i])]
    min_datum = np.array(min_datum)
    max_datum = np.array(max_datum)
    distance = max_datum - min_datum
    for i in range(len(data)):
        data[i] = np.array(data[i])
        data[i] = ((data[i] - min_datum) * 1.0 / distance).tolist()
        

 
def main():
    output_file = time.strftime("%H-%M-%S",time.localtime()) + '.txt'
    loss_out = open('./loss_' + output_file, 'w')
    acc_train_out = open('./acc_train_' + output_file, 'w')
    acc_test_out = open('./acc_test_' + output_file, 'w')


    data_file = './data.mat'
    data = scio.loadmat(data_file)

    sess = tf.InteractiveSession()

    in_units = 310
    h1_units = 200
    h2_units = 30
    h3_units = 10
    h4_units = 8
    out_units = 4

    batch_size = 5

    train_data = data['train_de']
    train_label = onehot(data['train_label_eeg'], out_units)
    test_data = data['test_de']
    test_label = onehot(data['test_label_eeg'], out_units)

    train_data = (train_data).tolist()
    test_data = (test_data).tolist()

    normalize(train_data, train_data+test_data)
    normalize(test_data, train_data+test_data)



    len_train_data = len(train_data)
    len_test_data = len(test_data)

    #print(train_data)

    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1))
    b2 = tf.Variable(tf.zeros([h2_units]))
    W3 = tf.Variable(tf.truncated_normal([h2_units, h3_units], stddev=0.1))
    b3 = tf.Variable(tf.zeros([h3_units]))
    W4 = tf.Variable(tf.truncated_normal([h3_units, h4_units], stddev=0.1))
    b4 = tf.Variable(tf.zeros([h4_units]))
    #W5 = tf.Variable(tf.truncated_normal([h4_units, out_units], stddev=0.1))
    #b5 = tf.Variable(tf.zeros([out_units]))

    W5 = tf.Variable(tf.truncated_normal([h3_units, out_units], stddev=0.1))
    b5 = tf.Variable(tf.zeros([out_units]))
    x = tf.placeholder(tf.float32, [None, in_units])
    keep_prob = tf.placeholder(tf.float32)
    
    hidden1 = tf.nn.relu(tf.matmul(x,W1) + b1)
    #hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    #hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, W2) + b2)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
    #hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
    hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + b3)
    hidden4 = tf.nn.relu(tf.matmul(hidden3, W4) + b4)
    #y = tf.nn.softmax(tf.matmul(hidden4, W5) + b5)
    y = tf.nn.softmax(tf.matmul(hidden3, W5) + b5)
    y_ = tf.placeholder(tf.float32, [None, out_units])
    regular = layers.l1_l2_regularizer(.5)(W1) + layers.l1_l2_regularizer(.5)(W2) + layers.l1_l2_regularizer(.5)(W3)
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)) + 0.0015 * regular)
    #loss = tf.reduce_mean(-tf.reduce_sum(tf.reduce_sum(y_ * tf.log(y)) + regular))
    #loss = cross_entropy + regular
    #train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    train_step = tf.train.AdagradOptimizer(0.001).minimize(loss)

    tf.global_variables_initializer().run()
    for j in range(0,500):
        for i in range(0, ma.floor(len_train_data / batch_size)):
            batch_x = next_batch(train_data, batch_size, i)
            batch_y = next_batch(train_label, batch_size, i)
            #batch_x = tf.train.batch(train_data, batch_size)
            #batch_y = tf.train.batch(train_label, batch_size)
            train_step.run({x:batch_x, y_:batch_y, keep_prob:0.75})
        print('Iter:' + str(j))
        total_cross_entropy = sess.run(loss, feed_dict={x:train_data, y_:train_label , keep_prob: 1.0})
        print('loss: ' + str(total_cross_entropy)) 
        loss_out.write(str(total_cross_entropy) + '\n')
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accur = accuracy.eval({x:train_data, y_:train_label, keep_prob: 1.0})
        test_accur = accuracy.eval({x:test_data, y_:test_label, keep_prob: 1.0})
        print(train_accur)
        print(test_accur)
        acc_train_out.write(str(train_accur) + '\n')
        acc_test_out.write(str(test_accur) + '\n')

    

main()







