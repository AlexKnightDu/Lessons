from svm import *
from svmutil import *

import multiprocessing as mp
import scipy.io as scio
import numpy as np
import os
import time

train_data_file = './splice_scale.txt'
test_data_file = './splice_scale.t'

kernel_type = ['linear', 'poly', 'rbf', 'sigmoid']

feature_num = 60

def train(kernel):
    time_stamp = time.strftime("%H-%M-%S",time.localtime())
    fout = open('./result_1/s' + kernel_type[kernel] + '_' + time_stamp + '.out', 'w+')
    print('kernel ' + kernel_type[kernel] + ' started ' + '*' * 20)
    print('the process parent id :',os.getppid())
    print('the process id is :',os.getpid())

    y, x = svm_read_problem(train_data_file)
    yt, xt = svm_read_problem(test_data_file)

    # Cost
    c_para = list(np.logspace(0,12,13,base=4)/1e4)
    g_para = list(np.logspace(0,6,7,base=2)/(8 * feature_num))
    d_para = list(range(2,7))
    r_para = list(range(0,3,1))
    for r in r_para:
        for g in g_para:
            g = round(g, 5)
            for c in c_para:
                c = round(c, 5)
                if (str(kernel) == '1'):
                    for d in d_para:
                        param = '-t ' + str(kernel) + ' -c ' + str(c) + ' -g ' + str(g) + ' -r ' + str(r) + ' -d ' + str(d)
                        fout.write(param + ' ')
                        param = svm_parameter(param + ' -b 1 -m 5000 -q')
                        problem = svm_problem(y,x)
                        model = svm_train(problem, param)
                        p_label, p_acc, p_val = svm_predict(yt, xt, model)
                        fout.write(str(p_acc) + '\n')
                        fout.flush()
                else:
                    param = '-t ' + str(kernel) + ' -c ' + str(c) + ' -g ' + str(g) + ' -r ' + str(r)
                    fout.write(param + ' ')
                    param = svm_parameter(param + ' -b 1 -m 5000 -q')
                    problem = svm_problem(y, x)
                    model = svm_train(problem, param)
                    p_label, p_acc, p_val = svm_predict(yt, xt, model)
                    fout.write(str(p_acc) + '\n')
                    fout.flush()
    fout.close()


def main():
    pool = mp.Pool()
    processes = []
    result = []

    kernel_num = 4

    # train(0)
    for i in range(kernel_num):
        processes += [pool.apply_async(train, args=(i,))]

    pool.close()
    pool.join()

    # for i in ran
    # ge(0, kernel_num):
        # temp = processes[i].get()
        # result += [temp]

    # print(result)

main()
