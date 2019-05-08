#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import h5py


# In[34]:


f = h5py.File('../data/book_events_total_view_2017-01-06.h5', 'r')
keys = list(f.keys())
pd.DataFrame(f[keys[0]][:]).head()


# In[20]:


sec = pd.read_csv('../data/securities.csv')
sec = sec[['Ticker symbol', 'GICS Sector']]
sec = sec.drop_duplicates()
sec.head()


# In[29]:


# path_to_file = './csv/2017-01-03/'
h5file_path = '../data/book_events_total_view_2017-01-06.h5'

csv_file_name = h5file_path.split('/')[2].split('.')[0]
csv_file_name


# In[32]:


start_time = time.time()
csv_path_to_file = './csv/2017-01-06/'
with h5py.File('../data/book_events_total_view_2017-01-06.h5', 'r') as f:
    final_df = pd.DataFrame()
    for key in list(f.keys()):
#         path_to_file = './csv/2017-01-03/'
#         key2 = key[1:]
        if key in list(sec['Ticker symbol']):
            print(key)
            df = pd.DataFrame(f[key][:])
            df = df[df['book_event_type']==b'T']
            # get sector
            sector = sec[sec['Ticker symbol'] == key]['GICS Sector'].iloc[0]
            df['sector'] = sector
            df['ticker'] = key
#             path_to_file += sector
#             if not os.path.exists(path_to_file):
#                 os.mkdir(path_to_file)
      

      final_df = final_df.append(df)
        
final_df.to_csv('./data/'+ csv_file_name +'.csv')

print("--- %s seconds ---" % (time.time() - start_time))

