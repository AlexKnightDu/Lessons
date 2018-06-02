from sklearn.decomposition import PCA
import numpy as np
from PIL import Image



def normalize(x):
    m=x.shape[0]
    n=x.shape[1] 
    t=x.shape[2]
    for i in range(m):
        for j in range(n):
            for k in range(t):
                if (x[i][j][k]>50): x[i][j][k] = 1
                else: x[i][j][k] = 0

x_train_data = np.fromfile("mnist_train_data",dtype=np.uint8)
y_train_label = np.fromfile("mnist_train_label",dtype=np.uint8)
x_test_data = np.fromfile("mnist_test_data",dtype=np.uint8)
y_test_label = np.fromfile("mnist_test_label",dtype=np.uint8)


print(x_train_data.shape)
print(y_train_label.shape)
print(x_test_data.shape)
print(y_test_label.shape)
fig_w = 45
new_fig_w = 45
#reshape the matrix
x_train_data = x_train_data.reshape(-1,fig_w,fig_w)
x_test_data = x_test_data.reshape(-1,fig_w,fig_w)
print("After reshape:",x_train_data.shape)
print("After reshape:",x_test_data.shape)
new_x_train_data = np.empty([60000,new_fig_w,new_fig_w])
new_x_test_data = np.empty([10000,new_fig_w,new_fig_w])
for i in range(60000):
    im = Image.fromarray(x_train_data[i])
    im.thumbnail((new_fig_w,new_fig_w))
    new_x_train_data[i] = np.array(im)
for i in range(10000):
    im = Image.fromarray(x_test_data[i])
    im.thumbnail((new_fig_w,new_fig_w))
    new_x_test_data[i] = np.array(im)
print(new_x_train_data.shape)
print(new_x_test_data.shape)
normalize(new_x_train_data)
normalize(new_x_test_data)
np.save('x_train_data.npy',new_x_train_data)
np.save('x_test_data.npy',new_x_test_data)



