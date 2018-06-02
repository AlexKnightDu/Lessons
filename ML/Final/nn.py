import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

fig_w = 20
x_train_data = np.load('x_train_data.npy')
x_test_data = np.load('x_test_data.npy')
#x_train_data = np.fromfile("mnist_train_data",dtype=np.uint8)
#x_test_data = np.fromfile("mnist_test_data",dtype=np.uint8)
x_train_data = x_train_data.reshape(-1,fig_w*fig_w)
x_test_data = x_test_data.reshape(-1,fig_w*fig_w)
x_train_data = x_train_data.astype(np.float64)
x_test_data = x_test_data.astype(np.float64)
x_train_data -= np.mean(x_train_data,axis = 0)
x_train_data /= np.std(x_train_data,axis = 0)
x_test_data -= np.mean(x_test_data,axis = 0)
x_test_data /= np.std(x_test_data,axis = 0)


y_train_label = np.fromfile("mnist_train_label",dtype=np.uint8)
y_test_label = np.fromfile("mnist_test_label",dtype=np.uint8)
lambd=0.9

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

# the layer function
def add_layer(inputs,insize,outsize,active_function=None):
    w = tf.Variable(tf.truncated_normal([insize,outsize], stddev=0.001))
    b = tf.Variable(tf.zeros([1,outsize]),dtype = tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(w))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(b))
    result = tf.add(tf.matmul(inputs,w),b)
    if active_function==None:
        return result
    else:
      result = active_function(result)
      return result

#to construct the graph

x = tf.placeholder(dtype = tf.float32,shape = [None,fig_w*fig_w])
y = tf.placeholder(dtype = tf.float32,shape = [None,10])

l1 = add_layer(x,fig_w*fig_w,60,tf.nn.relu)
prediction = add_layer(l1,60,10,tf.nn.softmax)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#loss =tf.reduce_mean(tf.square(y-prediction))
loss  = -tf.reduce_sum(y*tf.log(tf.clip_by_value(prediction, 1e-10, 1e100)))
tf.add_to_collection('losses', loss)
final_loss = tf.add_n(tf.get_collection('losses'))
#loss  = tf.losses.hinge_loss(logits = prediction, labels = y)
#loss =tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.AdamOptimizer (0.001).minimize(final_loss)
l = []
Time = []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(10000):
        x_batch,y_batch = get_batch_data()
        
        sess.run(train_step,feed_dict = {x:x_batch,y:y_batch})
        if i%50 ==0:
            print('loss:')
            print(sess.run(final_loss,feed_dict = {x:x_batch,y:y_batch}))
            print('test accuracy:')
            s =sess.run(accuracy,feed_dict = {x:x_test_data,y:y_test_label})
            print(s)
            print('train accuracy:')
            ss =sess.run(accuracy,feed_dict = {x:x_train_data,y:y_train_label})
            print(ss)
            l.append(s)
            
            
    
print(max(l))
index = l.index(max(l))

print(index)
plt.plot(l)
plt.show()

