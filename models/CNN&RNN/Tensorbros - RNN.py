#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[1]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.pipeline import Pipeline
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


# ## Data Pre-processing
# 
# We reshape the data for 1D convolution, with 5 features. We have variable **image_height**, which indicates the height of the convolution window. The default image height is 5.
# 

# In[27]:


# reshapes data into n (len(data)/image_height) image_heightx5 data values.
def split_reshape_dataset(X, y, image_height=1):
    
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
def make_dataset_whole(X, image_height=1):
    X = X.reset_index(drop='index')
    x_shape=X.shape[0]
    i = x_shape%image_height
    if i != 0:
        X = X.drop(list(range(x_shape-1, x_shape-i-1,-1)),axis=0)
    
    return X


# ## Split data into train and test

# In[28]:


data1 = pd.read_csv('../data/events_prices/events_prices/Consumer Discretionary.csv')


# In[29]:


y = data1.iloc[:, -1]
y.head()
X = data1.iloc[:, 1:6]
X.head()
X_train, X_test, y_train, y_test = split_reshape_dataset(X,y)

y_train.shape, X_train.shape


# In[6]:


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


# In[14]:


# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# In[30]:


def build_model(X_train, y_train):
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=50, batch_size=72, verbose=verbose)
    return model
# design network
# model = Sequential()
# model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')


# In[31]:


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


# In[35]:


np.mean(list(mse_dict.values()))


# In[22]:


# fit network
history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)


# In[23]:


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[24]:


yhat = model.predict(X_test)


# In[26]:


rmse = sqrt(mean_squared_error(y_test, yhat))
print('Test RMSE: %.3f' % rmse)


# In[ ]:


def build_model(X_train, y_train):
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(300, return_sequences = True))
    model.add(LSTM(800))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=50, batch_size=72, verbose=verbose)
    return model


# Financial -- Information Technology

# In[22]:


f_it = f.loc[f['target_sector'] == "Information Technology"]


# In[23]:


f_it.head()


# In[24]:


X = f_it[['height', 'width', 'distance', 'left_slope', 'right_slope']]
Y = f_it['target_sector_average_price']
X = X.values
Y = Y.values
#Test size?
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[25]:


# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# In[28]:


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


# In[29]:


# fit network
history = model.fit(X_train, Y_train, epochs=50, batch_size=72, validation_data=(X_test, Y_test), verbose=2, shuffle=False)


# In[30]:


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[31]:


yhat = model.predict(X_test)


# In[32]:


rmse = sqrt(mean_squared_error(Y_test, yhat))
print('Test RMSE: %.3f' % rmse)


# Health Care -- Financial

# In[36]:


hc_f = hc.loc[hc['target_sector'] == "Financials"]


# In[37]:


hc_f.head()


# In[38]:


X = hc_f[['height', 'width', 'distance', 'left_slope', 'right_slope']]
Y = hc_f['target_sector_average_price']
X = X.values
Y = Y.values
#Test size?
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[39]:


# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# In[40]:


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


# In[41]:


# fit network
history = model.fit(X_train, Y_train, epochs=50, batch_size=72, validation_data=(X_test, Y_test), verbose=2, shuffle=False)


# In[42]:


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[43]:


yhat = model.predict(X_test)


# In[44]:


rmse = sqrt(mean_squared_error(Y_test, yhat))
print('Test RMSE: %.3f' % rmse)


# Health Care -- Information Tech

# In[45]:


hc_it = hc.loc[hc['target_sector'] == "Information Technology"]


# In[46]:


hc_it.head()


# In[48]:


X = hc_it[['height', 'width', 'distance', 'left_slope', 'right_slope']]
Y = hc_it['target_sector_average_price']
X = X.values
Y = Y.values
#Test size?
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[49]:


# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# In[54]:


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


# In[55]:


# fit network
history = model.fit(X_train, Y_train, epochs=50, batch_size=72, validation_data=(X_test, Y_test), verbose=2, shuffle=False)


# In[56]:


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[57]:


yhat = model.predict(X_test)


# In[58]:


rmse = sqrt(mean_squared_error(Y_test, yhat))
print('Test RMSE: %.3f' % rmse)


# Information Tech -- Health Care

# In[62]:


it_hc = it.loc[it['target_sector'] == "Health Care"]


# In[63]:


it_hc.head()


# In[64]:


X = it_hc[['height', 'width', 'distance', 'left_slope', 'right_slope']]
Y = it_hc['target_sector_average_price']
X = X.values
Y = Y.values
#Test size?
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[65]:


# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# In[66]:


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


# In[67]:


# fit network
history = model.fit(X_train, Y_train, epochs=50, batch_size=72, validation_data=(X_test, Y_test), verbose=2, shuffle=False)


# In[68]:


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[69]:


yhat = model.predict(X_test)


# In[70]:


rmse = sqrt(mean_squared_error(Y_test, yhat))
print('Test RMSE: %.3f' % rmse)


# Information Tech -- Financial

# In[71]:


it_f = it.loc[it['target_sector'] == "Financials"]


# In[72]:


it_f.head()


# In[73]:


X = it_f[['height', 'width', 'distance', 'left_slope', 'right_slope']]
Y = it_f['target_sector_average_price']
X = X.values
Y = Y.values
#Test size?
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[74]:


# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# In[75]:


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


# In[76]:


# fit network
history = model.fit(X_train, Y_train, epochs=50, batch_size=72, validation_data=(X_test, Y_test), verbose=2, shuffle=False)


# In[77]:


# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[78]:


yhat = model.predict(X_test)


# In[79]:


rmse = sqrt(mean_squared_error(Y_test, yhat))
print('Test RMSE: %.3f' % rmse)

