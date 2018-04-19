#import pudb
#pu.db

import tensorflow as tf
import numpy as np
import scipy.io as scio
#import pandas as pd
import math as ma

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

 
def main():
    data_file = './data.mat'
    data = scio.loadmat(data_file)

    sess = tf.InteractiveSession()

    in_units = 310
    h1_units = 300
    h2_units = 200
    h3_units = 100
    h4_units = 10
    out_units = 4

    batch_size = 5

    train_data = data['train_de']
    train_label = onehot(data['train_label_eeg'], out_units)
    test_data = data['test_de']
    test_label = onehot(data['test_label_eeg'], out_units)

    len_train_data = len(train_data)
    len_test_data = len(test_data)

    train_data = tf.Variable(train_data, dtype=tf.float32)
    test_data = tf.Variable(test_data, dtype=tf.float32)

    axis = list(range(1))
    mean, variance = tf.nn.moments(train_data, axis)
    scale = tf.Variable(tf.ones([310]))
    offset = tf.Variable(tf.zeros([310]))
    variance_epsilon = 0.001
    train_data = tf.nn.batch_normalization(train_data, mean, scale, offset, scale, variance_epsilon)
    test_data = tf.nn.batch_normalization(test_data, mean, scale, offset, scale, variance_epsilon)


    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1))
    b2 = tf.Variable(tf.zeros([h2_units]))
    W3 = tf.Variable(tf.truncated_normal([h2_units, h3_units], stddev=0.1))
    b3 = tf.Variable(tf.zeros([h3_units]))
    W4 = tf.Variable(tf.truncated_normal([h3_units, h4_units], stddev=0.1))
    b4 = tf.Variable(tf.zeros([h4_units]))
    W5 = tf.Variable(tf.truncated_normal([h4_units, out_units], stddev=0.1))
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
    y = tf.nn.softmax(tf.matmul(hidden4, W5) + b5)
    y_ = tf.placeholder(tf.float32, [None, out_units])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
    #train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    train_step = tf.train.AdagradOptimizer(0.008).minimize(cross_entropy)

    tf.global_variables_initializer().run()
    for j in range(0,200):
        for i in range(0, ma.floor(len_train_data / batch_size)):
            batch_x = next_batch(train_data, batch_size, i)
            batch_y = next_batch(train_label, batch_size, i)
            train_step.run({x:batch_x, y_:batch_y, keep_prob:0.75})
            if i % 500 == 0:
                total_cross_entropy = sess.run(cross_entropy, feed_dict={x:train_data, y_:train_label , keep_prob: 1.0})
                print(str(i * batch_size) + 'steps: ' + str(total_cross_entropy)) 
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval({x:train_data, y_:train_label, keep_prob: 1.0}))

    

main()







