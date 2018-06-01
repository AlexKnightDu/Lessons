import glob
import time
from PIL import Image
from skimage.feature import hog
import os
import pickle
import numpy as np

from svm import *
from svmutil import *


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def preprocess_data():
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

    fx = get_feat(x[0:10],32)
    fxt = get_feat(xt[0:10],32)
    np.savetxt('train_feat.in', fx)
    np.savetxt('test_feat.in', fxt)
    np.savetxt('train_labels.in',y[0:10],fmt='%d')
    np.savetxt('test_labels.in',yt[0:10],fmt='%d')
    return fx,y,fxt,yt

def load_data():
    x = np.loadtxt('train_feat.in')
    xt = np.loadtxt('test_feat.in')
    y = np.loadtxt('train_labels.in',dtype='int')
    yt = np.loadtxt('test_labels.in',dtype='int')
    return x,y,xt,yt


def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

def transform(image):
    r = image[0:1024]
    g = image[1024:2048]
    b = image[2048:3072]
    img = np.column_stack((r,g,b))
    img = img.reshape(32,32,3)
    return img




def get_feat(images,size):
    feat_images = []
    for image in images:
        image = transform(image)
        gray = rgb2gray(image)/255.0
        fea = hog(gray, orientations=12, pixels_per_cell=[8,8], cells_per_block=[2,2], visualise=False, transform_sqrt=True)
        feat_images += [fea]
    return feat_images



def main():
    x,y,xt,yt = preprocess_data()
    print(x)
    print(y)
    x,y,xt,yt = load_data()
    print(x)
    print(y)

    print(type(x))
<<<<<<< HEAD



    # param = '-t 0 -c 4 -b 1'
    # param = svm_parameter(param)#+ ' -b 1 -m 5000 -q')
    # problem = svm_problem(y, x)
    # model = svm_train(problem, param)
    # p_label, p_acc, p_val = svm_predict(yt, xt, model)
    # print(p_acc)
=======
    fx = get_feat(x,32)
    fxt = get_feat(xt,32)

    param = '-t 0 -c 4 -b 1'
    param = svm_parameter(param)#+ ' -b 1 -m 5000 -q')
    problem = svm_problem(y, fx)
    model = svm_train(problem, param)
    p_label, p_acc, p_val = svm_predict(yt, fxt, model)
    print(p_acc)
>>>>>>> a3e8c42c0f193a6df4a96148da3679fc07a1fdd9
    # fout.write(str(p_acc) + '\n')
    # fout.flush()

main()


