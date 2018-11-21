#import pudb
#pu.db

import tensorflow as tf
import numpy as np
import scipy.io as scio
import math as ma
import multiprocessing as mp
from tensorflow.contrib import layers
import time
import os

def next_batch(data, label, batch_size):
    index  = np.arange(len(data))
    np.random.shuffle(index)
    index = index[0:batch_size]
    batch_data = data[index]
    batch_label = label[index]
    return batch_data,batch_label

def onehot(labels, units):
    #print(labels)
    l = len(labels)
    onehot_labels = np.zeros([l,units])
    for i in range(0,l):
        onehot_labels[i][int(labels[i])] = 1
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
        
def network(parameters):
    train_data = parameters[0]
    train_label = parameters[1]
    test_data = parameters[2]
    test_label = parameters[3]
    decri = parameters[4]
    descri = parameters[5]

    print('the process parent id :',os.getppid())  
    print('the process id is :',os.getpid())

    loss_out = open(descri + '_loss.txt', 'w')
    acc_train_out  = open(descri + '_acc_train.txt', 'w')

    #acc_test_out.write(str(test_accur) + '\n')

    in_units = 310
    h1_units = 40
    out_units = 2
    learning_rate = 0.0001
    regular_ratio = 0.9
    batch_num = 300
    batch_size = 100
    iter_num = 100

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

    all_prediction = []
    tf.global_variables_initializer().run()
    for j in range(0,iter_num):
        for i in range(0, batch_num):
            batch_x, batch_y = next_batch(train_data, train_label, batch_size)
            train_step.run({x:batch_x, y_:batch_y})

        result = tf.argmax(y,1)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        total_cross_entropy = sess.run(loss, feed_dict={x:train_data, y_:train_label})
        train_accur = accuracy.eval({x:train_data, y_:train_label})
        #test_accur = accuracy.eval({x:test_data, y_:test_label})
        print('Iter:' + str(j))
        print('loss: ' + str(total_cross_entropy)) 
        print(train_accur)
        loss_out.write(str(total_cross_entropy) + '\n')
        acc_train_out.write(str(train_accur) + '\n')
        prediction = (sess.run(result, feed_dict = {x:test_data}))
        all_prediction += [prediction]
    all_prediction = np.array(all_prediction)
    return [decri,all_prediction]


def process_data(train_data, train_label, random):
    data = [[],[],[],[]]
    labels = [[],[],[],[]]
    pair_data = [[],[],[],[]]
    pair_labels = [[],[],[],[]]
    for i in range(0,len(train_label)):
        data[train_label[i][0]] += [train_data[i]]
        labels[train_label[i][0]] += [1]
    if (random):
        for i in range(0,len(train_label)):
            for j in range(0,4):
                if (j != train_label[i][0]):
                    data[j] += [train_data[i]]
                    labels[j] += [0]
        for i in range(0,4):
            data[i] = np.array(data[i])
            labels[i] = np.array(labels[i])
        return data, labels
    else:
        for i in range(0,4):
            for j in range(0,4):
                if (i != j):
                    pair_data[i] += [np.array(data[i]+data[j])]
                    pair_labels[i] += [np.array(labels[i]+np.zeros(len(data[j])).tolist())]
        return pair_data, pair_labels
        
        
def prior_generate_data(train_data, train_label, min_num, max_num, num):
    data = []
    labels = []
    for i in range(0,max_num):
        data += [[]]
        labels += [[]]
        for j in range(0,min_num):
                datum, label = next_batch(train_data[j] , train_label[j], num)
                data[i] += [datum]
                labels[i] += [(label)]
    return np.array(data), np.array(labels)
    
def random_generate_data(train_data, train_label, min_num, max_num, num):
    data = []
    labels = []
    for i in range(0,max_num):
        data += [[]]
        labels += [[]]
        for j in range(0,min_num):
                datum, label = next_batch(train_data, train_label, num)
                data[i] += [datum]
                labels[i] += (label)
    return np.array(data), np.array(labels)

def minmax(results, min_num, max_num, test_label):
    min_result = []
    max_result = []
    for i in range(0,max_num):
        min_result += [[]]
        for j in range(0,len(results[0][0])):
            min_result[i] += [[]]
            for k in range(0,len(test_label)):
                min_result[i][j] += [min(results[i][:][:,j][:,k])]
    min_result = np.array(min_result)
    #print('________')
    #print(min_result)
    for i in range(0,len(min_result[0])):
        max_result += [[]]
        for j in range(0,len(test_label)):
            max_result[i] += [max(min_result[:,i][:,j])]
    return max_result
        

def main():
    min_num = 1
    max_num = 1
    cate_num = 4
    sub_data_size = 1000

    data_file = './data.mat'
    data = scio.loadmat(data_file)

    out_units = 4
    train_data = data['train_de']
    test_data = data['test_de']

    normalize(train_data, test_data)
    normalize(test_data, test_data)

    train_label = data['train_label_eeg']
    test_label = data['test_label_eeg']


    ovr_random_data, ovr_random_label = process_data(train_data, train_label, True)
    ovr_prior_data, ovr_prior_label = process_data(train_data, train_label, False)
    #ovr_test_data, ovr_test_label = process_data(test_data, test_label)
    for i in range(0,len(ovr_random_label)):
        ovr_random_label[i] = onehot(ovr_random_label[i], 2)
    #print(ovr_random_label)
    for i in range(0,max_num):
        for j in range(0,max_num-1):
            ovr_prior_label[i][j] = onehot(ovr_prior_label[i][j],2)
    train_label = np.concatenate(train_label)
    test_label = np.concatenate(test_label)
    train_label = onehot(data['train_label_eeg'], out_units)
    test_label = onehot(data['test_label_eeg'], out_units)

    results = [[],[],[],[]]

    time_stamp = time.strftime("%H-%M-%S",time.localtime()) 

    for k in range(0,cate_num):
        #M3_data, M3_label = random_generate_data(ovr_random_data[k], ovr_random_label[k], min_num, max_num, sub_data_size)
        M3_data, M3_label = prior_generate_data(ovr_prior_data[k], ovr_prior_label[k], min_num, max_num, sub_data_size)

        len_train_data = len(train_data)
        len_test_data = len(test_data)
    

        pool = mp.Pool()
        processes = []
        result = []
        for i in range(0, max_num):
            processes += [[]]
            result += [[]]
            for j in range(0, min_num):
                descri = './t_' + time_stamp + '_' + str(i) + '_' + str(j)
                parameters = [M3_data[i][j], M3_label[i][j], test_data, test_label, i, descri]
                processes[i] += [pool.apply_async(network, args=(parameters,))]
                result[i] += [[]]
        pool.close()
        pool.join()

        for i in range(0, max_num):
            for j in range(0, min_num):
                temp = processes[i][j].get()
                result[temp[0]][j] = temp[1]
        for i in range(0,max_num):
            result[i] = np.array(result[i])
        result = np.array(result)
        #print(result)

    
        #results[k] += [(result)]
        results[k] += minmax(result, min_num, max_num, test_label)

    #for k in range(cate_num):
        #print(results[k])

    final_result = []
    for w in range(0,len(results[0])):
        final_result += [[]] 
        for u in range(0,len(test_label)):
            flag = 1
            for v in range(cate_num):
                if results[v][w][u] == 1:
                    flag = 0
                    final_result[w] += [v]
                    break
            if flag:
                final_result[w] += [0]
    print(final_result)
    real = np.concatenate(data['test_label_eeg'])
    prediction = final_result

    pred_out = open('./p_' + time_stamp + '_predict.txt' , 'w')
    acc_out = open('./p_' + time_stamp + '_acc.txt' , 'w')
    for w in range(0,len(results[0])):
        prediction_static = []
        for i in range(4):
            prediction_static += [[0,0,0,0]]
        for i in range(0,len(real)):
            prediction_static[real[i]][prediction[w][i]] += 1
        for i in range(4):
            print(prediction_static[i])
        final_accuracy = 0
        for i in range(4):
            final_accuracy += prediction_static[i][i]
        final_accuracy = (final_accuracy * 1.0) / len(test_label)
        print(final_accuracy)
        pred_out.write(str(prediction_static) + '\n')
        acc_out.write(str(final_accuracy) + '\n')



main()




