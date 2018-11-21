import tensorflow as tf
import numpy as np
data_x1 = np.random.uniform(2,3,1000)
data_x2 = np.random.uniform(12,13,1000)
data_y1 = np.random.uniform(15,13,1000)
data_y2 = np.random.uniform(30,20,1000)

train_output = [] 
test_output = []
train_result_output = []
test_result_output = []

rands = np.random.rand(1000)
for i in range(700):
        if (rands[i] > 0.5):
                train_output.append([data_x1[i],data_y1[i]])
                train_result_output.append(1)
        else:
                train_output.append([data_x2[i],data_y2[i]])
                train_result_output.append(2)

for i in range(700,1000):
        if rands[i] > 0.5:
                test_output.append([data_x1[i],data_y1[i]])
                test_result_output.append(1)
        else:
                test_output.append([data_x2[i],data_y2[i]])
                test_result_output.append(2)


##################################################

learning_rate = 0.0001

p = tf.placeholder(tf.float32,[1,2])
t = tf.placeholder(tf.float32,[1])

W = tf.Variable(tf.zeros([1,2]),name="weights")
b = tf.Variable(tf.zeros([1]),name="bias")

a = tf.matmul(W,tf.transpose(p))+b

errors = []
error_output = open('./error_'+str(learning_rate)+'.txt','w+')

loss = tf.reduce_sum(tf.square(t - a))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for num_step in range(1000):
        error = 0
        for i in range(700):
                sess.run(train_step,feed_dict={p: [train_output[i]], t: [train_result_output[i]]})
        for i in range(300):
                error = error + sess.run(loss,feed_dict={p: [test_output[i]], t: [test_result_output[i]]})
        print(error)
        error_output.write(str(error) + '\n')
        errors.append(error)


error_output.close()
   






