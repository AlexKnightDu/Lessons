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
    num = np.random.randint(0,4)
    for i in range(num):
        x = np.random.randint(0,wid)
        scale_x = np.random.randint(1,6)
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


def main():
    print(os.getcwd())

    file1= './train-images-idx3-ubyte'
    file2= 'train-labels-idx1-ubyte'

    imgs,data_head = loadImageSet(file1)
    imgs = imgs[0:100]
    print('data_head:',data_head)
    print(type(imgs))
    print('imgs_array:',imgs)
    print(np.reshape(imgs[1,:],[28,28])) #取出其中一张图片的像素，转型为28*28，大致就能从图像上看出是几啦

    for i in range(len(imgs)):
        # img = 255 - imgs[i]
        img = imgs[i]
        for j in range(len(img)):
            if img[j] > 127:
                img[j] = 255
            else:
                img[j] = 0
        img = img.astype('uint8')
        imgs[i] = img
    np.save('mnist_reverse.npy',imgs)

    imgs = np.load('mnist_reverse.npy')
    print(imgs.shape)

    data = imgs
    fig_w = 28
    data_num = len(data)
    # data = data.reshape(data_num, fig_w, fig_w)

    print("After reshape:", data.shape)

    for i in range(10):
        # choose a random index
        ind = np.random.randint(0, data_num)

        # print the index and label
        ima = data[ind].reshape(fig_w,fig_w)
        ima = expand(ima, 28, 45)
        add_noise(ima,45)
        # save the figure
        im = Image.fromarray(ima)
        im.save("example" + str(i) + ".png")

    labels,labels_head = loadLabelSet(file2)
    print('labels_head:',labels_head)
    print(type(labels))
    print(labels)

main()