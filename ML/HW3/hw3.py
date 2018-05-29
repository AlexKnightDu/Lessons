from svm import *
from svmutil import *

import scipy.io as scio
import numpy as np

train_data_file = './a9a.txt'
test_data_file = './a9a.t'

y, x = svm_read_problem(train_data_file)
yt, xt = svm_read_problem(test_data_file)

param = svm_parameter('-t 0 -c 4 -b 1')
model = svm_train(y[0:1000], x[0:1000])
print('test:')
p_label, p_acc, p_val = svm_predict(yt[0:1000], xt[0:100], model)
print(p_label)
# 线性核
# param = svm_parameter('-t 0 -c 4 -b 1')
# RBF核
# param = svm_parameter('-t 2 -c 4 -b 1')
