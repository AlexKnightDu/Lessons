
# encoding: utf-8
import numpy as np
import struct
import os
from PIL import Image

def loadImageSet(filename):
    binfile = open(filename, 'rb') # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0) # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset) # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height]) # reshape为[60000,784]型数组
    imgs = imgs.astype('uint8')
    return imgs,head


def loadLabelSet(filename):
    binfile = open(filename, 'rb') # 读二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0) # 取label文件前2个整形数

    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    numString = '>' + str(labelNum) + "B" # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset) # 取label数据

    binfile.close()
    labels = np.reshape(labels, [labelNum]) # 转型为列表(一维数组)

    return labels,head

def expand(image,wid,new_wid):
    ima = np.zeros((new_wid,new_wid), dtype='uint8')
    p = np.random.randint(0,new_wid-wid)
    q = np.random.randint(0,new_wid-wid)
    for i in range(wid):
        for j in range(wid):
            ima[p+i][q+j] = image[i][j]
    return ima


def add_noise(image, wid):
    num = np.random.randint(0,5)
    for i in range(num):
        x = np.random.randint(0,wid)
        scale_x = np.random.randint(3,7)
        y = np.random.randint(0,wid)
        if (x + y) % 2 == 0:
            flag = 1
        else:
            flag = 0
        for i in range(x, min(x+scale_x,wid)):
            scale_y = np.random.randint(3, 5)
            for j in range(y, min(y+scale_y,wid)):
                if (flag):
                    image[i][j] = 255
                else:
                    image[j][i] = 255

def preprocess(data_file, label_file, out_file):
    imgs, data_head = loadImageSet(data_file)
    labels, labels_head = loadLabelSet(label_file)

    np.save(out_file + '_labels.npy', labels)
    print(out_file +' labels fininshed')
    # imgs = imgs[0:100]
    expand_imgs = []
    for i in range(len(imgs)):
        # img = 255 - imgs[i]
        img = imgs[i]
        for j in range(len(img)):
            if img[j] > 127:
                img[j] = 255
            else:
                img[j] = 0
        img = img.astype('uint8')
        img = img.reshape(28, 28)
        expand_imgs += [expand(img, 28, 45)]

    expand_imgs = np.array(expand_imgs, dtype='uint8')
    np.save(out_file + '_expand.npy', expand_imgs)
    print(out_file +' expand fininshed')
    for i in range(len(imgs)):
        add_noise(expand_imgs[i], 45)
    np.save(out_file + '_noise.npy', expand_imgs)
    print(out_file +' noise fininshed')

    # imgs = np.load('mnist_expand.npy')
    # n_imgs = np.load('mnist_noise.npy')

    # data = imgs
    # ndata = n_imgs
    # data_num = len(data)
    # for i in range(10):
    #     # choose a random index
    #     ind = np.random.randint(0, data_num)
    #
    #     ima = data[ind]
    #     nima = ndata[ind]  # .reshape(fig_w,fig_w)
    #     im = Image.fromarray(ima)
    #     im.save("example" + str(i) + ".png")
    #     nim = Image.fromarray(nima)
    #     nim.save("example" + str(i) + "n.png")


def main():
    train_data_file = './train-images-idx3-ubyte'
    train_label_file = './train-labels-idx1-ubyte'
    test_data_file = './t10k-images-idx3-ubyte'
    test_label_file = './t10k-labels-idx1-ubyte'
    preprocess(train_data_file,train_label_file,'train')
    preprocess(test_data_file,test_label_file,'test')
    #
    # imgs = np.load('train_expand.npy')
    # n_imgs = np.load('train_noise.npy')
    # labels = np.load('train_labels.npy')
    #
    # data = imgs
    # ndata = n_imgs
    # data_num = len(data)
    # for i in range(10):
    #     # choose a random index
    #     ind = np.random.randint(0, data_num)
    #
    #     ima = data[ind]
    #     nima = ndata[ind]  # .reshape(fig_w,fig_w)
    #     im = Image.fromarray(ima)
    #     im.save("example" + str(i) + ".png")
    #     nim = Image.fromarray(nima)
    #     nim.save("example" + str(i) + "n.png")
    # print(imgs)

main()