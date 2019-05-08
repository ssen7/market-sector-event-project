#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv('../data/event_data_correct/Health Care.csv')

data = data[data['target_sector']=='Financials']
data.name = 'Event_HC_Tar_Fin'

X = data.iloc[:, :5]
y = data.iloc[:, -1]


def split_reshape_dataset(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    X_train = make_dataset_whole(X_train)
    X_test = make_dataset_whole(X_test)
    y_train = make_dataset_whole(y_train)
    y_test = make_dataset_whole(y_test)
    
    X_train = np.reshape(X_train.values, (-1,5,5))
    X_test = np.reshape(X_test.values, (-1,5,5))
    y_train = np.reshape(y_train.values, (-1,5))
    y_test = np.reshape(y_test.values, (-1,5))
    
    return X_train, X_test, y_train, y_test

def make_dataset_whole(X):
    X = X.reset_index(drop='index')
    x_shape=X.shape[0]
    i = x_shape%5
    if i != 0:
        X = X.drop(list(range(x_shape-1, x_shape-i-1,-1)),axis=0)
    
    return X

X_train, X_test, y_train, y_test = split_reshape_dataset(X,y)

y_train.shape, X_train.shape


# In[3]:


height = 5
width = 5
# channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

pool3_dropout_rate = 0.25
pool3_fmaps = conv2_fmaps

n_fc1 = 128
fc1_dropout_rate = 0.5
n_outputs=y_train.shape[1]
n_outputs


# In[4]:


# Step 2: Set up placeholders for input data
with tf.name_scope("inputs"):
    X_ = tf.placeholder(tf.float32, shape=[None,height, width], name="X_")
#     X_reshaped = tf.reshape(X_, shape=[-1, height, width])
    y_ = tf.placeholder(tf.int32, shape=[None,n_outputs], name="y_")
    training = tf.placeholder_with_default(False, shape=[], name='training')


# In[5]:


# Step 3: Set up the two convolutional layers using tf.layers.conv1d
conv1 = tf.layers.conv1d(X_, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv1d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")


# In[6]:


# Step 4: Set up the pooling layer with dropout using tf.layers.max_pooling1d
with tf.name_scope("pool3"):
    pool3 = tf.layers.max_pooling1d(conv2, pool_size=2, strides=1)
    pool3_flat = tf.contrib.layers.flatten(pool3)
#     pool3_flat_drop = tf.layers.dropout(pool3_flat, pool3_dropout_rate, training=training)


# In[7]:


with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)


# In[8]:


n_outputs=y_train.shape[1]
with tf.name_scope("output"):
    output = tf.layers.dense(fc1, n_outputs, name="output")
    


# In[9]:


output.shape


# In[10]:


with tf.name_scope("train"):
#     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.losses.mean_squared_error(y_, output)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)


# In[11]:


with tf.name_scope("eval"):
    mse = tf.reduce_mean(tf.squared_difference(tf.cast(y_, tf.float32), output))


# In[12]:


with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# In[13]:


# Step 9: Define some necessary functions
def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        print()
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


# In[14]:


print("scroll down for final test mse")
n_epochs = 10
batch_size = 4
iteration = 0

best_loss_val = np.infty
check_interval = 5
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None 

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            iteration += 1
            sess.run(training_op, feed_dict={X_: X_batch, y_: y_batch, training: True})
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X_: X_test, y_: y_test})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
#         if checks_since_last_progress > max_checks_without_progress:
#             print("Early stopping!")
#             break

    if best_model_params:
        restore_model_params(best_model_params)
    mse_test = mse.eval(feed_dict={X_: X_test, y_: y_test})
    print("Final MSE on test set:", mse_test)
    save_path = saver.save(sess, "./my1d_convolution_model")


# In[ ]:




