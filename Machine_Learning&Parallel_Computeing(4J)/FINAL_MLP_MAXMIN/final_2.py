#import pudb
#pu.db

import tensorflow as tf
import numpy as np
import scipy.io as scio
#import pandas as pd
import math as ma
import multiprocessing as mp
from tensorflow.contrib import layers
import time
import os

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
        
class MLP(mp.Process):
    def __init__(self, train_data, train_label, test_data, test_label, result_queue, descri):
        mp.Process.__init__(self)
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.descri = descri
        self.result_queue = result_queue

    def run(self):
        print('the process parent id :',os.getppid())  
        print('the process id is :',os.getpid())

        in_units = 310
        h1_units = 40
        out_units = 4
        learning_rate = 0.0001
        regular_ratio = 0.9
        batch_num = 300
        batch_size = 100
        iter_num = 5

        sess = tf.InteractiveSession()

        W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
        b1 = tf.Variable(tf.zeros([h1_units]))
        W5 = tf.Variable(tf.truncated_normal([h1_units, out_units], stddev=0.1))
        b5 = tf.Variable(tf.zeros([out_units]))

        x = tf.placeholder(tf.float32, [None, in_units])
    
        hidden1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
        y = tf.nn.softmax(tf.matmul(hidden1, W5) + b5)
        y_ = tf.placeholder(tf.float32, [None, out_units])

        regular = layers.l2_regularizer(.5)(W1) + layers.l2_regularizer(.5)(W5) 
        loss = -tf.reduce_sum(y_ * tf.log(y)) + regular_ratio  * regular

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        tf.global_variables_initializer().run()
        for j in range(0,iter_num):
            for i in range(0, batch_num):
                batch_x, batch_y = next_batch(self.train_data, self.train_label, batch_size)
                train_step.run({x:batch_x, y_:batch_y})

            result = tf.argmax(y,1)
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) 
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            total_cross_entropy = sess.run(loss, feed_dict={x:self.train_data, y_:self.train_label})
            train_accur = accuracy.eval({x:self.train_data, y_:self.train_label})
            test_accur = accuracy.eval({x:self.test_data, y_:self.test_label})
            print('Iter:' + str(j))
            print('loss: ' + str(total_cross_entropy)) 
            print(train_accur)
            print(test_accur)
        prediction = (sess.run(result, feed_dict = {x:self.test_data}))
        real = (sess.run(result, feed_dict = {y:self.test_label}))
        print(prediction)
        print(real)
        self.descri = 1
        self.prediction = prediction
        self.get()


def process_data(train_data, train_label):
    data = [[],[],[],[]]
    labels = [[],[],[],[]]
    for i in range(0,len(train_label)):
        data[train_label[i][0]] += [train_data[i]]
        labels[train_label[i][0]] += [[1]]
        for j in range(0,4):
            if (j != train_label[i][0]):
                data[j] += [train_data[i]]
                labels[j] += [[-1]]
    for i in range(0,4):
        data[i] = np.array(data[i])
        labels[i] = np.array(labels[i])
    return data, labels

def random_generate_data(train_data, train_label, min_num, max_num, num):
    data = []
    labels = []
    index  = np.arange(min_num * max_num)
    np.random.shuffle(index)
    for i in range(0,max_num):
        for j in range(0,min_num):
                datum, label = next_batch(train_data[index[i*min_num + j] % 4], train_label[index[i*min_num + j] % 4], num)
                data += [datum]
                labels += [label]
    return data, labels
    

def prior_generate_data(train_data, train_label, min_num, max_num, num):
    data = []
    label = []
    index  = np.arange(max_num)
    np.random.shuffle(index)
    for i in range(0,max_num):
        for j in range(0,min_num):
                datum, label = next_batch(train_data[index[i] % 4], train_label[index[i] % 4], num)
                data += [datum]
                labels += [label]
    return data, labels

def main():
    min_num = 3
    max_num = 4
    sub_data_size = 10

    data_file = './data.mat'
    data = scio.loadmat(data_file)

    out_units = 4
    train_data = data['train_de']
    test_data = data['test_de']

    normalize(train_data, test_data)
    normalize(test_data, test_data)

    train_label = data['train_label_eeg']
    ovr_data, ovr_label = process_data(train_data, train_label)
    for labels in ovr_label:
        labels = onehot(labels, out_units)
    train_label = onehot(data['train_label_eeg'], out_units)
    test_label = onehot(data['test_label_eeg'], out_units)


    M3_data, M3_label = random_generate_data(ovr_data, ovr_label, min_num, max_num, sub_data_size)
    #M3_data, M3_label = prior_generate_data(ovr_data, ovr_label, min_num, max_num, sub_data_size)

    
    len_train_data = len(train_data)
    len_test_data = len(test_data)
    

    processes = []
    results = []
    result_queue = mp.Queue()
    for i in range(0, max_num):
        processes += [[]]
        results += [[]]
        for j in range(0, min_num):
            processes[i] += [MLP(train_data, train_label, test_data, test_label, result_queue, 'test')]
            results[i] += [[]]

    for i in range(0, max_num):
        for j in range(0, min_num):
            processes[i][j].start()
    for i in range(0, max_num):
        for j in range(0, min_num):
            processes[i][j].join()

    for i in range(0, max_num):
        for j in range(0, min_num):
            results[i][j] = processes[i][j].get()
    print(results)

    #output_file = descri + '_' + time.strftime("%H-%M-%S",time.localtime()) + '.txt'
    #loss_out = open('./loss_' + output_file, 'w')
    #acc_train_out = open('./acc_train_' + output_file, 'w')
    #acc_test_out = open('./acc_test_' + output_file, 'w')
    #pred_out = open('./predict_' + output_file, 'w')

    #acc_train_out.write(str(train_accur) + '\n')
    #acc_test_out.write(str(test_accur) + '\n')

    #prediction_static = []
    #for i in range(4):
        #prediction_static += [[0,0,0,0]]
    #for i in range(0,len(real)):
        #prediction_static[real[i]][prediction[i]] += 1
    #for i in range(4):
        #print(prediction_static[i])
    #pred_out.write(str(prediction_static))
    #loss_out.write(str(total_cross_entropy) + '\n')


main()




