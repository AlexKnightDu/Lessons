import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression

fig_w = 28
x_train_data = np.load('x_train_data.npy')
x_test_data = np.load('x_test_data.npy')
#x_train_data = np.fromfile("mnist_train_data",dtype=np.uint8)
#x_test_data = np.fromfile("mnist_test_data",dtype=np.uint8)

x_train_data = x_train_data.reshape(-1,fig_w*fig_w)
x_test_data = x_test_data.reshape(-1,fig_w*fig_w)
x_train_data = x_train_data.astype(np.float32)
x_test_data = x_test_data.astype(np.float32)
x_train_data -= np.mean(x_train_data,axis = 0)
x_train_data /= np.std(x_train_data,axis = 0)
x_test_data -= np.mean(x_test_data,axis = 0)
x_test_data /= np.std(x_test_data,axis = 0)

y_train_label = np.fromfile("mnist_train_label",dtype=np.uint8)
y_test_label = np.fromfile("mnist_test_label",dtype=np.uint8)
tf.reset_default_graph()
#get one-hot vector
def get_new_label(y):
    m = y.shape[0]
    result = np.zeros([m,10])
    result[np.arange(m),y] = 1
    return result
y_train_label = get_new_label(y_train_label)
y_test_label = get_new_label(y_test_label)

#get batch data

def get_batch_data(batchSize=1000):
    a = np.arange(60000)
    np.random.shuffle(a)
    index = a[0:batchSize]
    x_batch_data = x_train_data[index,:]
    y_batch_data = y_train_label[index]
    return x_batch_data,y_batch_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, fig_w*fig_w])
y = tf.placeholder(tf.float32, [None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,fig_w,fig_w,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)




W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)



cross_entropy =-tf.reduce_sum(y*tf.log(tf.clip_by_value(y_conv, 1e-10, 1e100)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(20000):
        batch_xs, batch_ys = get_batch_data(500)
        
        sess.run([train_step], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            s = sess.run(accuracy, feed_dict={
            x: x_test_data,
            y: y_test_label,
        })
            print('test:',s)
            l.append(s)
            print('train:',sess.run(accuracy, feed_dict={
            x: x_train_data,
            y: y_train_label,
        }))
'''
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(10000):
        batch_xs, batch_ys = get_batch_data(100)
        sess.run(h_pool1, feed_dict={
            x: batch_xs,
            y: batch_ys,
        })

        if i % 20 == 0:
            s = sess.run(cross_entropy, feed_dict={
            x: x_train_data,
            y: y_train_label,
        })
            print('test:',s)




