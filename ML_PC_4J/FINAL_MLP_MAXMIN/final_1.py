import tensorflow as tf
import numpy as np
import scipy.io as scio
import panda as pd
import math as ma

def main():
    data_file = './data.mat'
    data = scio.loadmat(data_file)

    train_data = data['train_de']
    train_label = data['train_label_eeg']
    test_data = data['test_de']
    test_label = data['test_label_eeg']

    sess = tf.InteractiveSession()

    in_units = 310
    h1_units = 200
    h2_units = 200
    out_units = 4

    batch_size = 100

    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    W2 = tf.Variable(tf.zeros([h1_units, h2_units]))
    b2 = tf.Variable(tf.zeros([h2_units]))
    W3 = tf.Variable(tf.zeros([h2_units, out_units]))
    b3 = tf.Variable(tf.zeros([out_units]))

    x = tf.placeholder(tf.float32, [None, in_units])
    keep_prob = tf.placeholder(tf.float32)
    
    hidden1 = tf.nn.relu(tf.matmul(x,W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, W2) + b2)
    hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
    y = tf.nn.softmax(tf.matmul(hidden2_frop, W3) + b3)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indicices=[1]))
    train_step = tf.train.AdagradDAOptimizer(0.3).minimize(cross_entropy)

    tf.global_variables_initializer().run()
    for i in range(0, ma.floor(len(train_data) / batch_size)):
        batch_x = next_batch(train_data, i)
        batch_y = next_batch(train_label, i)
        train_step.run({x: batch_x, y_:batch_y, keep_prob:0.75})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval({x:test_data, y_:test_label, keep_prob: 1.0}))

    


   
    


main()







