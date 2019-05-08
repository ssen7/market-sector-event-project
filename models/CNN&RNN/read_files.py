#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import os
import time
import matplotlib.pyplot as plt


# ## Get sector data from Kaggle
# https://www.kaggle.com/dgawlik/nyse#securities.csv
# 

# In[3]:


sec = pd.read_csv('./securities.csv')
sec.head()


# ### We only need the unique tickers and the GICS Sector
# 
# Ref: https://www.fidelity.com/learning-center/trading-investing/markets-sectors/global-industry-classification-standard
# 

# In[30]:


sec = sec[['Ticker symbol', 'GICS Sector']]
sec = sec.drop_duplicates()
sec.head()
# list(sec['Ticker symbol'])
pd.unique(sec['GICS Sector'])


# ### We need to make directories for each sector

# In[56]:


sec.shape


# In[49]:


sec['GICS Sector'].value_counts()


# In[38]:


# path_to_file = './csv/2017-01-03/'
h5file_path = './data/book_events_total_view_2017-01-03.h5'

csv_file_name = h5file_path.split('/')[2].split('.')[0]


# ### We store one day's worth of data into on csv file with the appropriate ticker and sector given for each traded stock

# In[42]:


start_time = time.time()
csv_path_to_file = './csv/2017-01-03/'
with pd.HDFStore(h5file_path, 'r') as train:
    final_df = pd.DataFrame()
    for key in train.keys():
#         path_to_file = './csv/2017-01-03/'
        key2 = key[1:]
        if key2 in list(sec['Ticker symbol']):
            df = train.get(key)
            df = df[df['book_event_type']=='T']
            # get sector
            sector = sec[sec['Ticker symbol'] == key2]['GICS Sector'].iloc[0]
            df['sector'] = sector
            df['ticker'] = key2
#             path_to_file += sector
#             if not os.path.exists(path_to_file):
#                 os.mkdir(path_to_file)
            final_df = final_df.append(df)
            
    final_df.to_csv(csv_path_to_file + '/' + csv_file_name +'.csv')

print("--- %s seconds ---" % (time.time() - start_time))


# ### Data sanity check

# In[53]:


sanity = final_df[['sector', 'ticker']].drop_duplicates()


# In[57]:


print(sanity.shape == sec.shape)

