from svm import *
from svmutil import *

import multiprocessing as mp
import scipy.io as scio
import numpy as np
import os
import time

train_data_file = './protein'
test_data_file = './protein.t'

kernel_type = ['linear', 'poly', 'rbf', 'sigmoid']

feature_num = 357

def svm_read_problem(data_file_name):
    prob_y = []
    prob_x = []
    for line in open(data_file_name):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        xi = {}
        # print(features)
        features = features.split()
        i = 0
        while i < len(features):
            ind, val = features[i].split(":")
            # print(features[i])
            # print(val == "")
            # print(float(val))
            if val == "":
                val = features[i + 1]
                i = i + 1
            # print(val)
            xi[int(ind)] = float(val)
            i = i + 1
        prob_y += [float(label)]
        prob_x += [xi]
    print('Finished')
    return (prob_y, prob_x)


def train(kernel):
    time_stamp = time.strftime("%H-%M-%S",time.localtime())
    fout = open('./result_2/' + kernel_type[kernel] + '_' + time_stamp + '.out', 'w+')
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
