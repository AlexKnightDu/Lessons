#import pudb
#pu.db


from sklearn.neural_network import MLPClassifier as mlpc
from sklearn import datasets as ds
from sklearn.metrics import classification_report,confusion_matrix

import time
import scipy.sparse as sps
import numpy as np

train_data_file = './a2a.txt'
test_data_file = './a2a.t'



def load_data():
    x,y=ds.load_svmlight_file(train_data_file)
    xt,yt=ds.load_svmlight_file(test_data_file)
    x.todense()
    xt.todense()
    x = sps.csr_matrix((x.data, x.indices, x.indptr), shape=(x.get_shape()[0], xt.get_shape()[1]))

    return x,y,xt,yt


def generate_data():
    fin = open('a9a.txt', 'r')
    fout = open('aba.txt', 'w')
    pos = 0
    neg = 0
    i = 0
    while ((pos < 1000) or (neg < 900)):
        line = fin.readline()
        y = line.split(' ')
        if (int(y[0]) > 0):
            if (pos < 1000):
                fout.write(line)
                pos += 1
        else:
            if ((neg < 900) or (np.random.randint(0, 100) % 20 == 0)):
                fout.write(line)
                neg += 1
        i += 1
    fin.close()
    fout.close()


generate_data()


