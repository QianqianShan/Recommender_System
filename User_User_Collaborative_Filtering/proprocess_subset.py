#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import pickle 
import numpy as np 
import pandas as pd 
from collections import Counter 

# read the edited ratings file by preprocess.py
os.chdir(r'/home/qshan/Desktop/Repos/Big_Data_Projects/Recommender_System/User_User_Collaborative_Filtering')
df = pd.read_csv('../data/ml-20m/edited_ratings.csv')
df.describe()


# In[2]:


print('Original data size: ', len(df))


# In[5]:


# number of occurences of userId and movieId 
user_ids_count = Counter(df.userId)
movie_ids_count = Counter(df.movie_idx)


# In[10]:


user_ids_count.most_common(5)


# In[11]:


# number of users and movies to keep in an subset 
n = 10000
m = 2000

# find the most common userId and movieId 
user_ids = [u_id for u_id, _ in user_ids_count.most_common(n)]
movie_ids = [m_id for m_id, _ in movie_ids_count.most_common(m)]

# deep copy 
df_small = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()


# reindex the user and movie ids so they are consecutive 
new_user_id_map = {}
count = 0 
for id_ in user_ids:
    new_user_id_map[id_] = count 
    count += 1 
    
new_movie_id_map = {}
count = 0 
for id_ in movie_ids:
    new_movie_id_map[id_] = count 
    count += 1 

print('Add new ids back to data')
df_small.loc[:, 'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId], axis = 1)
df_small.loc[:, 'movie_idx'] = df_small.apply(lambda row: new_movie_id_map[row.movie_idx], axis = 1)
print('New ids added')


# In[12]:


df_small.describe()


# In[13]:


# save to disk 
df_small.to_csv('../data/ml-20m/small_ratings.csv', index = False)

