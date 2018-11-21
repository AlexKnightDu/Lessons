from svm import *
from svmutil import *

import scipy.io as scio

def check(a,b,c,e,f,g):
    if (a == 1): return 1
    if (b == 1): return 0
    if (c == 1): return -1
    if ((e >= f) and (e >= g)): return 1
    if ((f >= e) and (f >= g)): return 0
    if ((g >= e) and (g >= f)): return -1
    return 1

def confusion_matrix(predict,test):
    matrix = [[0,0,0],[0,0,0],[0,0,0]]
    predict = list(map(lambda x:int(x), predict))
    test = list(map(lambda x:int(x), test))
    for i in range(len(test)):
        if (predict[i] == test[i]):
            matrix[predict[i]+1][predict[i]+1] += 1
        else:
            matrix[test[i]+1][predict[i]+1] += 1
    print("Predict: \t 1 \t 0 \t -1 ")
    print("Real 1: \t " + str(matrix[0][0]) + '\t' + str(matrix[0][1]) + '\t' + str(matrix[0][2]))
    print("Real 0: \t " + str(matrix[1][0]) + '\t' + str(matrix[1][1]) + '\t' + str(matrix[1][2]))
    print("Real -1: \t " + str(matrix[2][0]) + '\t' + str(matrix[2][1]) + '\t' + str(matrix[2][2]))
    return matrix


def main():
    train_data_file = './EEG_emotion_3/train_data.mat'
    test_data_file = './EEG_emotion_3/test_data.mat'
    train_label_file = './EEG_emotion_3/train_label.mat'
    test_label_file = './EEG_emotion_3/test_label.mat'

    train_data = scio.loadmat(train_data_file)['train_data']
    test_data = scio.loadmat(test_data_file)['test_data']
    train_label = scio.loadmat(train_label_file)['train_label']
    test_label = scio.loadmat(test_label_file)['test_label']

    train_data = list(map(lambda x:dict(zip(range(310), x)), train_data))
    test_data = list(map(lambda x:dict(zip(range(310),x)), test_data))
    train_label = list(map(lambda x:x[0],train_label))
    test_label = list(map(lambda x:x[0],test_label))


    # 线性核
    linear_param = svm_parameter('-t 0 -c 1 -b 1 -g 1')
    # RBF核
    RBF_param = svm_parameter('-t 2 -c 1 -b 1 -g 0.0001')


    # One versus One
    prob = svm_problem(train_label, train_data)
    OO_linear_model = svm_train(prob, linear_param)
    OO_RBF_model = svm_train(prob, RBF_param)
    OO_linear_predict = svm_predict(test_label, test_data, OO_linear_model)
    OO_RBF_predict = svm_predict(test_label, test_data, OO_RBF_model)


    print("###############################################")


    # One versus Rest
    train_label_1 = list(map(lambda x:-1 if x != 1 else 1, train_label))
    train_label_0 = list(map(lambda x:-1 if x != 0 else 1, train_label))
    train_label__1 = list(map(lambda x:-1 if x != -1 else 1, train_label))
    prob_1 = svm_problem(train_label_1, train_data)
    prob_0 = svm_problem(train_label_0, train_data)
    prob__1 = svm_problem(train_label__1, train_data)


    # linear
    OR_linear_1_model = svm_train(prob_1, linear_param)
    OR_linear_0_model = svm_train(prob_0, linear_param)
    OR_linear__1_model = svm_train(prob__1, linear_param)

    OR_linear_1_predict = svm_predict(test_label, test_data, OR_linear_1_model)
    OR_linear_0_predict = svm_predict(test_label, test_data, OR_linear_0_model)
    OR_linear__1_predict = svm_predict(test_label, test_data, OR_linear__1_model)

    # RBF
    OR_RBF_1_model = svm_train(prob_1, RBF_param)
    OR_RBF_0_model = svm_train(prob_0, RBF_param)
    OR_RBF__1_model = svm_train(prob__1, RBF_param)

    OR_RBF_1_predict = svm_predict(test_label, test_data, OR_RBF_1_model)
    OR_RBF_0_predict = svm_predict(test_label, test_data, OR_RBF_0_model)
    OR_RBF__1_predict = svm_predict(test_label, test_data, OR_RBF__1_model)

    OR_linear_predict = []
    OR_RBF_predict = []

    for i in range(len(test_label)):
        OR_linear_predict += [check(OR_linear_1_predict[0][i], OR_linear_0_predict[0][i], OR_linear__1_predict[0][i], OR_linear_1_predict[2][i], OR_linear_0_predict[2][i], OR_linear__1_predict[2][i])]
        OR_RBF_predict += [check(OR_RBF_1_predict[0][i], OR_RBF_0_predict[0][i], OR_RBF__1_predict[0][i],OR_RBF_1_predict[2][i], OR_RBF_0_predict[2][i], OR_RBF__1_predict[2][i])]



    print("One versus one:")
    print("linear kernel:" + str(evaluations(test_label, OO_linear_predict[0])))
    confusion_matrix(OO_linear_predict[0], test_label)
    print("RBF kernel:" + str(evaluations(test_label, OO_RBF_predict[0])))
    confusion_matrix(OO_RBF_predict[0], test_label)
    print("One versus rest:")
    print("linear kernel:" + str(evaluations(test_label, OR_linear_predict)))
    confusion_matrix(OR_linear_predict, test_label)
    print("RBF kernel:" + str(evaluations(test_label, OR_RBF_predict)))
    confusion_matrix(OR_RBF_predict, test_label)

main()
