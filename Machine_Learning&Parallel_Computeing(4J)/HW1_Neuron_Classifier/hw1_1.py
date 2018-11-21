from numpy import *

def hardlims(x):
    if x >= 0:
        return 1
    else:
        return 0

def multiple(l1,l2):
    result = mat(l1)*mat(l2).T
    return result[0,0]

def train(train_data,W,b):
    flag = 0
    for train_datum in train_data:
        t = train_datum[1]
        a = predict(train_datum[0],W,b)
        e = t - a
        flag = flag + abs(e)
        W = map(lambda x,y: x + y, W, map(lambda x: e*x, train_datum[0]))
        b = b + e
    return W,b,flag

def test(test_data, W,b):
    for test_datum in test_data:
        a = predict(test_datum[0],W,b)
        test_datum[1] = a


def predict(data,W,b):
    return hardlims(multiple(W,data) + b)


def main():
    W = [0,0]
    b = 0
    train_data = [[[1,-1],1],[[-1,-1],1],[[0,0],0],[[1,0],0]]
    test_data = [[[-2,0],-1],[[1,1],-1],[[0,1],-1],[[-1,-2],-1]]
    iteration_num = 0
    flag = 1
    while(flag):
        flag = 0
        W,b,flag = train(train_data,W,b)
        test(test_data,W,b)
        print 'Iteration No: ', iteration_num
        print '*** W: ', W, '; b:' ,b
        print '*** test data and result: ', test_data
        iteration_num = iteration_num + 1
        
main()

              


