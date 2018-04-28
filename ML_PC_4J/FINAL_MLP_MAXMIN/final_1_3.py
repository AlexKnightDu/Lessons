#import pudb
#pu.db

import tensorflow as tf
import numpy as np
import scipy.io as scio
#import pandas as pd
import math as ma
from tensorflow.contrib import layers
import time

def next_batch1(data, label, batch_size):
    size = len(data)
    begin = np.random.randint(0,size)
    interval = np.random.randint(0,200)
    batch_data = []
    batch_label = []
    for i in range(0,batch_size):
        batch_data += [data[(begin + i * interval) % size]]
        batch_label += [label[(begin + i * interval) % size]]
    return batch_data, batch_label

def next_batch(data, label, batch_size):
    index  = np.arange(7293)
    np.random.shuffle(index)
    index = index[0:batch_size]
    batch_data = data[index]
    batch_label = label[index]
    return batch_data,batch_label


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
    medium_datum = (max_datum + min_datum) * 1.0 / 2
    distance = (max_datum - min_datum) * 1.0 / 2
    for i in range(len(data)):
        data[i] = np.array(data[i])
        data[i] = ((data[i] - medium_datum) / distance)
        
def main():
    output_file = time.strftime("%H-%M-%S",time.localtime()) + '.txt'
    loss_out = open('./loss_' + output_file, 'w')
    acc_train_out = open('./acc_train_' + output_file, 'w')
    acc_test_out = open('./acc_test_' + output_file, 'w')
    pred_out = open('./predict_' + output_file, 'w')

    data_file = './data.mat'
    data = scio.loadmat(data_file)
    sess = tf.InteractiveSession()

    in_units = 310
    h1_units = 40
    out_units = 4

    learning_rate = 0.0001
    regular_ratio = 0.9

    batch_num = 300
    batch_size = 100

    train_data = data['train_de']
    train_label = onehot(data['train_label_eeg'], out_units)
    test_data = data['test_de']
    test_label = onehot(data['test_label_eeg'], out_units)

    normalize(train_data, test_data)
    normalize(test_data, test_data)
 

    len_train_data = len(train_data)
    len_test_data = len(test_data)

    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    W5 = tf.Variable(tf.truncated_normal([h1_units, out_units], stddev=0.1))
    b5 = tf.Variable(tf.zeros([out_units]))

    x = tf.placeholder(tf.float32, [None, in_units])
    
    hidden1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
    y = tf.nn.softmax(tf.matmul(hidden1, W5) + b5)
    y_ = tf.placeholder(tf.float32, [None, out_units])

    regular = layers.l2_regularizer(.5)(W1) + layers.l2_regularizer(.5)(W5) #+ layers.l2_regularizer(.5)(W5)
    loss = -tf.reduce_sum(y_ * tf.log(y)) + regular_ratio  * regular

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    tf.global_variables_initializer().run()
    begin = time.time()
    for j in range(0,100):
        for i in range(0, batch_num):
            batch_x, batch_y = next_batch(train_data, train_label, batch_size)
            train_step.run({x:batch_x, y_:batch_y})


        print('Iter:' + str(j))
        total_cross_entropy = sess.run(loss, feed_dict={x:train_data, y_:train_label})
        print('loss: ' + str(total_cross_entropy)) 
        loss_out.write(str(total_cross_entropy) + '\n')
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        result = tf.argmax(y,1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accur = accuracy.eval({x:train_data, y_:train_label})
        test_accur = accuracy.eval({x:test_data, y_:test_label})
        print(train_accur)
        print(test_accur)

        acc_train_out.write(str(train_accur) + '\n')
        acc_test_out.write(str(test_accur) + '\n')

    end = time.time()
    print((end - begin))
    prediction = (sess.run(result, feed_dict = {x:test_data}))
    real = (sess.run(result, feed_dict = {y:test_label}))
    print(prediction)
    print(real)
    prediction_static = []
    for i in range(4):
        prediction_static += [[0,0,0,0]]
    for i in range(0,len(real)):
        prediction_static[real[i]][prediction[i]] += 1
    for i in range(4):
            print(prediction_static[i])
    pred_out.write(str(prediction_static))

    
main()




