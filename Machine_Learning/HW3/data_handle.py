
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def b(s):
    return bytes(s, encoding="utf8")

def load_data():
    x = None
    y = []
    for i in range(5):
        train_file_name = './cifar-10-batches-py/data_batch_' + str(i+1)
        dict = unpickle(train_file_name)
        if x is None:
            x = dict[b'data']
        else:
            x = np.row_stack((x,dict[b'data']))
        y += dict[b'labels']
    test_file_name = './cifar-10-batches-py/test_batch'
    dict = unpickle(test_file_name)
    xt = dict[b'data']
    yt = dict[b'labels']
    return x,y,xt,yt

def main():
    x,y,xt,yt = load_data()
    print(type(x))
main()

