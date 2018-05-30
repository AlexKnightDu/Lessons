# import pudb
# pu.db

import numpy as np
import scipy.io as scio
import math as ma


def next_batch(data, batch_size, i):
    s = len(data)
    if (i + 1) * batch_size > s:
        return data[i * batch_size:-1]
    else:
        return data[i * batch_size:(i + 1) * batch_size]


def onehot(labels, units):
    l = len(labels)
    onehot_labels = np.zeros([l, units])
    for i in range(0, l):
        onehot_labels[i][labels[i][0]] = 1
    return onehot_labels


def normalize(data, min_datum, max_datum):
    distance = max_datum - min_datum
    for i in range(len(data)):
        data[i] = (data[i] - min_datum) * 1.0 / distance


def preprocess(data):
    result = []
    return result


def main():
    train_data_file = './a9a.txt'
    test_data_file = './a9a.t'

    train_data = preprocess(np.loadtxt(train_data_file))
    test_data = preprocess(np.loadtxt(test_data_file))


    in_units = 123
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

    min_datum = min(train_data)
    max_datum = max(train_data)
    normalize(train_data, min_datum, max_datum)
    normalize(test_data, min_datum, max_datum)

    len_train_data = len(train_data)
    len_test_data = len(test_data)

    # train_data = np.array(train_data)#, dtype=np.float32)
    # test_data = np.array(test_data)#, dtype=np.float32)
    # train_data = tf.Variable(train_data, dtype=tf.float32)
    # test_data = tf.Variable(test_data, dtype=tf.float32)

    axis = list(range(1))
    variance_epsilon = 0.001

    print(train_data)



    # hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    # hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, W2) + b2)
    # hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
    # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)


main()







