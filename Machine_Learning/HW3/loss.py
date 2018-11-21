#import pudb
#pu.db


from sklearn.neural_network import MLPClassifier as mlpc
from sklearn import datasets as ds
from sklearn.metrics import classification_report,confusion_matrix

import time
import scipy.sparse as sps
import numpy as np





def generate_data():
    fin = open('splice.t', 'r')
    fout = open('splice_scale_loss.t', 'w')
    while 1:
        line = fin.readline()
        if not line:
            break
        y = line
        if (np.random.randint(0, 100) % 2 == 0):
            y = y.replace(str(np.random.randint(1, 5)) + '.000000', '0.0')
        y = y.replace('2.000000', '-0.333333')
        y = y.replace('3.000000', '0.333333')
        y = y.replace('1.000000', '-1')
        y = y.replace('4.000000', '1')

        fout.write(y)

    fin.close()
    fout.close()


generate_data()


