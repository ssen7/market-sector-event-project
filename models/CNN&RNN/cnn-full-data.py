#!/usr/bin/env python
# coding: utf-8

# In[60]:


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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from time import time
from tensorflow.python.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir='./final_logs/{}'.format(time()))


# ## Reading in the data

# In[2]:


data = pd.read_csv('../data/events_prices/events_prices/Consumer Discretionary.csv')
data.name = 'ConsumerDiscretion'
data.head(5)


# In[12]:


X = data.iloc[:, 1:6]
X.head()


# In[13]:


y = data.iloc[:, -1]
y.head()


# ## Data Pre-processing
# 
# We reshape the data for 1D convolution, with 5 features. We have variable **image_height**, which indicates the height of the convolution window. The default image height is 5.
# 

# In[39]:


# reshapes data into n (len(data)/image_height) image_heightx5 data values.
def split_reshape_dataset(X, y, image_height=5):
    
    X = X.sample(frac=0.05, random_state=100)
    y = y.sample(frac=0.05, random_state=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    X_train = make_dataset_whole(X_train,image_height)
    X_test = make_dataset_whole(X_test,image_height)
    y_train = make_dataset_whole(y_train,image_height)
    y_test = make_dataset_whole(y_test,image_height)
    
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
    x_shape=X.shape[0]
    i = x_shape%image_height
    if i != 0:
        X = X.drop(list(range(x_shape-1, x_shape-i-1,-1)),axis=0)
    
    return X


# ## Split data into train and test

# In[40]:


X_train, X_test, y_train, y_test = split_reshape_dataset(X,y)

y_train.shape, X_train.shape


# In[8]:


data1 = pd.read_csv('../data/events_prices/events_prices/Consumer Discretionary.csv')


# In[41]:


def get_target_dfs(data):
    df_list = []
    event = str(data.event_sector[0])
    for target in pd.unique(data.target_sector):
        
        target_name = str(target)
        df = data[data.target_sector == target]
        df.name = event+'-'+target_name
        df_list.append(df)
        
    return df_list

target_dflist = get_target_dfs(data1)


# In[7]:





# ## Build basic model using 1 1D convolution layer with kernel_size = 3

# In[42]:


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


# In[43]:


i = 0
# target_dflist = [data1, data2, data3, data4, data5, data6]
index = []
cols = []
mse_dict = {}
while i < len(target_dflist):
    data = target_dflist[i]
    name = data.name
    event_sector = name.split('-')[0]
    target_sector = name.split('-')[1]
    index+=[event_sector]
    cols+=[target_sector]
    X = data.iloc[:, 1:6]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_reshape_dataset(X,y)
    model = build_model(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_dict[name] = mse
    print(mse)
    i+=1
print(mse_dict)


# In[45]:


# consumer_dict = pd.DataFrame(mse_dict)
# consumer_dict.to_csv('CD_model1.csv')
np.mean(list(mse_dict.values()))


# ## Model with 2 convolution layers with kernel_size=2

# In[48]:


def build_model_2(X_train, y_train):
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    # define model
    model = Sequential()
    model.add(Conv1D(filters=20, kernel_size=2, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=10, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=verbose)
    return model


# In[49]:


i = 0
# target_dflist = [data1, data2, data3, data4, data5, data6]
index = []
cols = []
mse_dict = {}
while i < len(target_dflist):
    data = target_dflist[i]
    name = data.name
    event_sector = name.split('-')[0]
    target_sector = name.split('-')[1]
    index+=[event_sector]
    cols+=[target_sector]
    X = data.iloc[:, 1:6]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_reshape_dataset(X,y)
    model = build_model_2(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_dict[name] = mse
    print(mse)
    i+=1
print(mse_dict)


# ## Changing the timestep size from 5 to 10 and building model with 3 convolution layers.
# 
# ## Increasing the timestep allows us to use more convolution layers.

# In[50]:


def build_model_3(X_train, y_train):
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    # define model
    model = Sequential()
    
    model.add(Conv1D(filters=20, kernel_size=3, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=10, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=5, kernel_size=3, activation='relu'))
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

# In[53]:


i = 0
# target_dflist = [data1, data2, data3, data4, data5, data6]
index = []
cols = []
mse_dict = {}
while i < len(target_dflist):
    data = target_dflist[i]
    name = data.name
    event_sector = name.split('-')[0]
    target_sector = name.split('-')[1]
    index+=[event_sector]
    cols+=[target_sector]
    X = data.iloc[:, 1:6]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_reshape_dataset(X,y, image_height=10) ## change image height
    model = build_model_2(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_dict[name] = mse
    print(mse)
    i+=1
print(mse_dict)
np.mean(list(mse_dict.values()))


# In[54]:


## Final CNN
## Testing with build_model_3
i = 0
# target_dflist = [data1, data2, data3, data4, data5, data6]
index = []
cols = []
mse_dict = {}
while i < len(target_dflist):
    data = target_dflist[i]
    name = data.name
    event_sector = name.split('-')[0]
    target_sector = name.split('-')[1]
    index+=[event_sector]
    cols+=[target_sector]
    X = data.iloc[:, 1:6]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_reshape_dataset(X,y, image_height=10) ## change image height
    model = build_model_3(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_dict[name] = mse
    print(mse)
    i+=1
print(mse_dict)
np.mean(list(mse_dict.values()))


# In[56]:


# Using model 3 to build the rest of the models


# In[57]:


data2 = pd.read_csv('../data/events_prices/events_prices/Health Care.csv')
data3 = pd.read_csv('../data/events_prices/events_prices/Industrials.csv')
data4 = pd.read_csv('../data/events_prices/events_prices/Information Technology.csv')
data5 = pd.read_csv('../data/events_prices/events_prices/Consumer Staples.csv')
data6 = pd.read_csv('../data/events_prices/events_prices/Utilities.csv')
data7 = pd.read_csv('../data/events_prices/events_prices/Financials.csv')
data8 = pd.read_csv('../data/events_prices/events_prices/Real Estate.csv')
data9 = pd.read_csv('../data/events_prices/events_prices/Materials.csv')
data10 = pd.read_csv('../data/events_prices/events_prices/Energy.csv')
data11 = pd.read_csv('../data/events_prices/events_prices/Telecommunications Services.csv')


# In[ ]:


dataList = [data2, data3, data4, data5, data6, data7, data8, data9, data10, data11]
mse_list = []

for df in dataList:
    target_dflist = get_target_dfs(df)
    i = 0
    # target_dflist = [data1, data2, data3, data4, data5, data6]
    index = []
    cols = []
    mse_dict = {}
    while i < len(target_dflist):
        data = target_dflist[i]
        name = data.name
        event_sector = name.split('-')[0]
        target_sector = name.split('-')[1]
        index+=[event_sector]
        cols+=[target_sector]
        X = data.iloc[:, 1:6]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = split_reshape_dataset(X,y, image_height=10) ## change image height
        model = build_model_2(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_dict[name] = mse
        print(mse)
        i+=1
    print(mse_dict)
    mse_list.append(mse_dict)
#     np.mean(list(mse_dict.values()))
    
    

