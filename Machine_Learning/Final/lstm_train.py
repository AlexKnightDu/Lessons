import tensorflow as tf
import scipy.io as sio
import numpy as np



#get one-hot vector
def get_new_label(y):
    m = y.shape[0]
    result = np.zeros([m,10])
    result[np.arange(m),y] = 1
    return result


#get batch data

def get_batch_data(batchSize=1000):
    a = np.arange(60000)
    np.random.shuffle(a)
    index = a[0:batchSize]
    x_batch_data = x_train_data[index,:]
    y_batch_data = y_train_label[index]
    return x_batch_data,y_batch_data

#RNN layer
def RNN(X, weights, biases,num):
    #reshape X to 2 D
    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.9)(weights['in']))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.9)(biases['in']))

    # reshape X to 3 D
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # use BasicLSTM cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True,activation=tf.nn.relu)
    
    init_state = lstm_cell.zero_state(num, dtype=tf.float32) 
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.9)(weights['out']))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.9)(biases['out']))
    return results




tf.set_random_seed(1)   # set random seed

# to load data
#1898,0    2119,1    1573,2    1703,3    
fig_w = 30
x_train_data = np.load('x_train_data.npy')
x_test_data = np.load('x_test_data.npy')
#x_train_data = np.fromfile("mnist_train_data",dtype=np.uint8)
#x_test_data = np.fromfile("mnist_test_data",dtype=np.uint8)
x_train_data = x_train_data.reshape(-1,fig_w,fig_w)
x_test_data = x_test_data.reshape(-1,fig_w,fig_w)
x_train_data = x_train_data.astype(np.float64)
x_test_data = x_test_data.astype(np.float64)
x_train_data -= np.mean(x_train_data,axis = 0)
x_train_data /= np.std(x_train_data,axis = 0)
x_test_data -= np.mean(x_test_data,axis = 0)
x_test_data /= np.std(x_test_data,axis = 0)


y_train_label = np.fromfile("mnist_train_label",dtype=np.uint8)
y_test_label = np.fromfile("mnist_test_label",dtype=np.uint8)

y_train_label = get_new_label(y_train_label)
y_test_label = get_new_label(y_test_label)


# hyperparameters
lr = 0.001                  # learning rate
training_iters = 3000     # train step 
batch_size = 500            
n_inputs = fig_w              
n_steps = fig_w                # time steps
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              #the number of classes

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# to init weights
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}



pred = RNN(x, weights, biases,tf.shape(x)[0])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
tf.add_to_collection('losses', cost)
loss = tf.add_n(tf.get_collection('losses'))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
l = []
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step  < training_iters:
        batch_xs, batch_ys = get_batch_data(batch_size)
        
        sess.run([train_op], feed_dict={
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
            
        step += 1

print(max(l))
print(l.index(max(l)))
