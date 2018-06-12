from svm import *
from svmutil import *

import multiprocessing as mp
import scipy.io as scio
import numpy as np
import os
import time
from sklearn.metrics import classification_report,confusion_matrix

# train_data_file = './dna.scale.txt'
# test_data_file = './dna.scale.t'
train_data_file = './a9a.txt'
test_data_file = './a9a.t'


feature_num = 123

def train(kernel):

    time_stamp = time.strftime("%H-%M-%S",time.localtime())
    fout = open('./R' + '_' + time_stamp + '.out', 'w+')

    y, x = svm_read_problem(train_data_file)
    yt, xt = svm_read_problem(test_data_file)
    param = '-t 1 -c 0.4096 -g 0.01626 -r 1 -d 6 '
    fout.write(param + ' ')
    param = svm_parameter(param + ' -b 1 -m 1000 -h 1 -q')
    problem = svm_problem(y,x)
    model = svm_train(problem, param)
    p_label, p_acc, p_val = svm_predict(yt, xt, model)
    r = classification_report(yt, p_label)
    print(r)
    fout.write(str(r) + '\n')
    fout.flush()
    fout.close()


def main():
    pool = mp.Pool()
    processes = []
    result = []

    kernel_num = 4

    train(0)
    # for i in range(kernel_num):
    #     processes += [pool.apply_async(train, args=(i,))]
    #
    # pool.close()
    # pool.join()

    # for i in range(0, kernel_num):
        # temp = processes[i].get()
        # result += [temp]

    # print(result)

main()
