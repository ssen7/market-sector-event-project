#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from time import time
from tensorflow.python.keras.callbacks import TensorBoard

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout


tensorboard = TensorBoard(log_dir='./logs/{}'.format(time()))


# ## Reading in the data

# In[2]:


data = pd.read_csv('../data/event_data_correct/Health Care.csv')
data.name = 'Event_IT_Tar_Fin'
data.head(5)


# In[3]:


X = data.iloc[:, :5]
X.head()


# In[4]:


y = data.iloc[:, -1]
y.head()


# ## Data Pre-processing
#
# We reshape the data for 1D convolution, with 5 features. We have variable **image_height**, which indicates the height of the convolution window. The default image height is 5.
#

# In[5]:


# reshapes data into n (len(data)/image_height) image_heightx5 data values.
def split_reshape_dataset(X, y, image_height=5):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    X_train = make_dataset_whole(X_train, image_height)
    X_test = make_dataset_whole(X_test, image_height)
    y_train = make_dataset_whole(y_train, image_height)
    y_test = make_dataset_whole(y_test, image_height)

    X_train = np.reshape(X_train.values, (-1, image_height, 5))
    X_test = np.reshape(X_test.values, (-1, image_height, 5))
    y_train = np.reshape(y_train.values, (-1, image_height))
    y_test = np.reshape(y_test.values, (-1, image_height))

    return X_train, X_test, y_train, y_test

# to make sure we have uniform sized images.
# eg. If there are 161 rows of observations and we want to make predictions on 5 time steps each,
# this will reduce the dataset to 160 observations as 160%5==0


def make_dataset_whole(X, image_height=5):
    X = X.reset_index(drop='index')
    x_shape = X.shape[0]
    i = x_shape % image_height
    if i != 0:
        X = X.drop(list(range(x_shape-1, x_shape-i-1, -1)), axis=0)

    return X


# ## Split data into train and test

# In[6]:


X_train, X_test, y_train, y_test = split_reshape_dataset(X, y)

y_train.shape, X_train.shape


# In[7]:


data1 = pd.read_csv('../data/event_data_correct/Health Care.csv')
data2 = data1[data1['target_sector'] == 'Financials']
data1 = data1[data1['target_sector'] == 'Information Technology']
data1.name = 'HC_IT'
data2.name = 'HC_Fin'

data3 = pd.read_csv('../data/event_data_correct/Financials.csv')
data4 = data3[data3['target_sector'] == 'Health Care']
data3 = data3[data3['target_sector'] == 'Information Technology']
data3.name = 'Fin_IT'
data4.name = 'Fin_HC'

data5 = pd.read_csv('../data/event_data_correct/Information Technology.csv')
data6 = data5[data5['target_sector'] == 'Health Care']
data5 = data5[data5['target_sector'] == 'Financials']
data5.name = 'IT_Fin'
data6.name = 'IT_HC'


# ## Build basic model using 1 1D convolution layer with kernel_size = 3

# In[8]:


def build_model(X_train, y_train):
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu',
                     input_shape=(n_timesteps, n_features)))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=verbose)
    return model


# In[18]:


i = 0
datalist = [data1, data2, data3, data4, data5, data6]
index = []
cols = []
mse_dict = {}
while i < len(datalist):
    data = datalist[i]
    name = data.name
    event_sector = name.split('_')[0]
    target_sector = name.split('_')[1]
    index += [event_sector]
    cols += [target_sector]
    X = data.iloc[:, :5]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_reshape_dataset(X, y)
    model = build_model(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_dict[name] = mse
    print(mse)
    i += 1
print(mse_dict)


# In[23]:


np.mean(list(mse_dict.values()))


# ## Model with 2 convolution layers with kernel_size=2

# In[10]:


def build_model_2(X_train, y_train):
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    # define model
    model = Sequential()
    model.add(Conv1D(filters=10, kernel_size=2, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=20, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=verbose)
    return model


# In[24]:


i = 0
datalist = [data1, data2, data3, data4, data5, data6]
index = []
cols = []
mse_dict = {}
while i < len(datalist):
    data = datalist[i]
    name = data.name
    event_sector = name.split('_')[0]
    target_sector = name.split('_')[1]
    index += [event_sector]
    cols += [target_sector]
    X = data.iloc[:, :5]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_reshape_dataset(X, y)
    model = build_model_2(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_dict[name] = mse
    print(mse)
    i += 1
print(mse_dict)
np.mean(list(mse_dict.values()))


# ## Changing the timestep size from 5 to 10 and building model with 3 convolution layers.
#
# ## Increasing the timestep allows us to use more convolution layers.

# In[25]:


def build_model_3(X_train, y_train):
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    # define model
    model = Sequential()

    model.add(Conv1D(filters=10, kernel_size=3, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=20, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=20, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=verbose, callbacks=[tensorboard])
    return model


# ### Testing the changed timestep size with build_model 2

# In[26]:


i = 0
datalist = [data1, data2, data3, data4, data5, data6]
index = []
cols = []
mse_dict = {}
while i < len(datalist):
    data = datalist[i]
    name = data.name
    event_sector = name.split('_')[0]
    target_sector = name.split('_')[1]
    index += [event_sector]
    cols += [target_sector]
    X = data.iloc[:, :5]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_reshape_dataset(
        X, y, image_height=10)  # change image height
    model = build_model_2(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_dict[name] = mse
    print(mse)
    i += 1
print(mse_dict)
np.mean(list(mse_dict.values()))


# In[28]:


# Final CNN
# Testing with build_model_3
i = 0
datalist = [data1, data2, data3, data4, data5, data6]
index = []
cols = []
mse_dict = {}
while i < len(datalist):
    data = datalist[i]
    name = data.name
    event_sector = name.split('_')[0]
    target_sector = name.split('_')[1]
    index += [event_sector]
    cols += [target_sector]
    X = data.iloc[:, :5]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_reshape_dataset(
        X, y, image_height=10)  # change image height
    model = build_model_3(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_dict[name] = mse
    print(mse)
    i += 1
print(mse_dict)
np.mean(list(mse_dict.values()))


# ## Using Stochastic Gradient Descent optimizers

# In[29]:


def build_model_4(X_train, y_train):
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    # define model
    model = Sequential()
    model.add(Conv1D(filters=10, kernel_size=3, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=20, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=20, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='sgd')
    # fit network
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=verbose)
    return model


# In[31]:


# Testing with sgd
i = 0
datalist = [data1, data2, data3, data4, data5, data6]
index = []
cols = []
mse_dict = {}
while i < len(datalist):
    data = datalist[i]
    name = data.name
    event_sector = name.split('_')[0]
    target_sector = name.split('_')[1]
    index += [event_sector]
    cols += [target_sector]
    X = data.iloc[:, :5]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_reshape_dataset(
        X, y, image_height=10)  # change image height
    model = build_model_4(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_dict[name] = mse
    print(mse)
    i += 1
print(mse_dict)
np.mean(list(mse_dict.values()))
