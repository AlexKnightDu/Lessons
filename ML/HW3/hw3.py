from svm import *
from svmutil import *

import multiprocessing as mp
import scipy.io as scio
import numpy as np
import os
import time

train_data_file = './a9a.txt'
test_data_file = './a9a.t'

kernel_type = ['linear', 'poly', 'rbf', 'sigmoid']

feature_num = 123

def train(kernel):

    time_stamp = time.strftime("%H-%M-%S",time.localtime())
    fout = open('./result/' + kernel_type[kernel] + '_' + time_stamp + '.out', 'w+')
    print('kernel ' + kernel_type[kernel] + ' started ' + '*' * 20)
    print('the process parent id :',os.getppid())
    print('the process id is :',os.getpid())

    y, x = svm_read_problem(train_data_file)
    yt, xt = svm_read_problem(test_data_file)

    # Cost
    c_para = list(np.logspace(0,10,11,base=4)/1e4)
    g_para = list(np.logspace(0,6,7,base=2)/(8 * feature_num))
    d_para = list(range(2,7))
    r_para = list(range(0,3,1))
    for c in c_para:
        c = round(c, 5)
        for g in g_para:
            g = round(g, 5)
            for r in r_para:
                if (kernel == 1):
                    for d in d_para:
                        d = d_para
                        param = '-t ' + str(kernel) + ' -c ' + str(c) + ' -g ' + str(g) + ' -r ' + str(r) + ' -d ' + str(d)
                        fout.write(param + ' ')
                        param = svm_parameter(param + ' -b 1 -m 100')
                        problem = svm_problem(y,x)
                        model = svm_train(problem, param)
                        p_label, p_acc, p_val = svm_predict(yt, xt, model)
                        fout.write(str(p_acc) + '\n')
                else:
                    param = '-t ' + str(kernel) + ' -c ' + str(c) + ' -g ' + str(g) + ' -r ' + str(r)
                    fout.write(param + ' ')
                    param = svm_parameter(param + ' -b 1 -m 2000')
                    problem = svm_problem(y[0:10], x[0:10])
                    model = svm_train(problem, param)
                    p_label, p_acc, p_val = svm_predict(yt[0:10], xt[0:10], model)
                    fout.write(str(p_acc) + '\n')
    fout.close()


def main():
    pool = mp.Pool()
    processes = []
    result = []

    kernel_num = 4

    for i in range(kernel_num):
        processes += [pool.apply_async(train, args=(i,))]

    pool.close()
    pool.join()

    # for i in range(0, kernel_num):
        # temp = processes[i].get()
        # result += [temp]

    # print(result)

main()