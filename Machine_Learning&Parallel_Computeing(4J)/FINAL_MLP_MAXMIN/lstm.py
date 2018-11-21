import numpy as np
import scipy.io as sio
import tensorflow as tf
# CLASS_NUM=4
# BATCH_SIZE=13
# HIDDEN_SIZE=128
# LEARNING_RATE=1e-8
# KEEP_PROB=0.5
# 500

CLASS_NUM=4
BATCH_SIZE=40
HIDDEN_SIZE=128
LEARNING_RATE=1e-11
KEEP_PROB=0.5

data = sio.loadmat('./data.mat')

def read_data(data,labels,target_length):
    data_size=labels.shape[0]
    formated_datas=[]
    formated_labels=[]
    masks=[]

    start=0
    end=0
    while end<data_size:
        start,end=get_next_range(labels,start)

        formated_data, mask = truncate_or_pad_data(data[start:end + 1, :], target_length)
        one_hot_label = np.zeros(CLASS_NUM)
        one_hot_label[labels[start, 0]] = 1

        formated_datas.append(formated_data)
        formated_labels.append(one_hot_label)
        masks.append(mask)

        start=end
    return np.array(formated_datas),np.array(masks), np.array(formated_labels)

def get_next_range(labels,start):
    start_pos_label=labels[start,0]
    end=start

    length=labels.shape[0]
    while end<length:
        if labels[end,0] == start_pos_label:
            end+=1
        else:
            break

    return start,end


def truncate_or_pad_data(data, truncate_length):
    # mask=np.zeros(truncate_length)
    for i in range(data.shape[0]):
        ele_min,ele_max=np.min(data[i,:]),np.max(data[i,:])
        data[i,:]=(data[i,:]-ele_min)/(ele_max-ele_min)
    mask=truncate_length-1
    if data.shape[0]>=truncate_length:
        processed_data= data[:truncate_length, :]
    else:
        mask=data.shape[0]-1
        processed_data=np.append(data, np.zeros((truncate_length - data.shape[0], data.shape[1])), axis=0)

    return processed_data,mask

train_data, train_masks,train_label = read_data(data['train_de'], data['train_label_eeg'],80)
test_data, test_masks,test_label = read_data(data['test_de'], data['test_label_eeg'],80)

def next_batch(data,mask,label,batch_index,batch_size):
    length=data.shape[0]
    batch_index=int(batch_index%(length/batch_size))

    data_=[]
    label_=[]
    for i in range(batch_index*batch_size,(batch_index+1)*batch_size):
        data_.append(data[i%length,:,:])
        label_.append(label[i%length,:])
    mask=[[i,mask[i%length]] for i in range(batch_size)]

    return  np.array(data_),mask,np.array(label_)



x=tf.placeholder(tf.float32,[None,80,310])
y = tf.placeholder(tf.float32, [None, CLASS_NUM])
mask=tf.placeholder(tf.int32,[None,2])
keep_prob = tf.placeholder(tf.float32)

lstm_cell=tf.nn.rnn_cell.LSTMCell(num_units=HIDDEN_SIZE,forget_bias=1.0,state_is_tuple=True)
lstm_cell=tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,input_keep_prob=1.0,output_keep_prob=keep_prob)
init_state=lstm_cell.zero_state(BATCH_SIZE,dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs=x, initial_state=init_state, time_major=False)
h_state = tf.gather_nd(outputs,mask)        # shape: (BATCH_SIZE, HIDDEN_SIZE)
# h_state=outputs[:,-1,:]

W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, CLASS_NUM], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[CLASS_NUM]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)     # shape: (batch_size,CLASS_NUM)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pre,labels=y))
train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(150):
        batch= next_batch(train_data,train_masks,train_label,i,BATCH_SIZE)
        if (i+1)%200 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], mask:batch[1],y: batch[2],keep_prob:KEEP_PROB})
            # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
            print ("step %d, training accuracy %g" % ( (i+1), train_accuracy))

        sess.run(train_op, feed_dict={x:batch[0], mask:batch[1],y: batch[2],keep_prob:KEEP_PROB})

    avg_acc=[]
    for i in range(int(test_data.shape[0]/BATCH_SIZE+0.5)):
        batch = next_batch(test_data, test_masks, test_label, i, BATCH_SIZE)
        acc=sess.run(accuracy, feed_dict={ x:batch[0], mask: batch[1], y: batch[2],keep_prob: 1.0})
        avg_acc.append(acc)
        print("test accuracy %g"% acc)

print(np.mean(avg_acc))