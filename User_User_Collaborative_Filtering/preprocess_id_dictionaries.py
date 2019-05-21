#!/usr/bin/env python
# coding: utf-8

# This script preprocess the data to three dictionaries: 
# 
# produce user: rated movies,
# 
# movie:users rated this movie, 
# 
# (user, movie):rating

# In[3]:


import os 
import pickle 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 

# load data 
os.chdir(r'/home/qshan/Desktop/Repos/Big_Data_Projects/Recommender_System/User_User_Collaborative_Filtering')
df = pd.read_csv('../data/ml-20m/small_ratings.csv')


# In[4]:


# randomly shuffle the data to create train/test data sets 
df = shuffle(df)
cutoff = int(0.8 * len(df))
df_train, df_test = df.iloc[ :cutoff], df.iloc[cutoff: ]


# In[5]:


# create dictionary to show which movies (value) are rated by which user(key)
user_movie = {}

# create dictionary to show which users (value) have rated which movie(key)
movie_user = {}

# create dictionary to find ratings for a user and a movie
usermovie_rating = {}


# In[6]:


print('Update dictionaries for train set')
def update_user_movie(row):
    # extract user and movie ids 
    user_id = int(row.userId)
    movie_id = int(row.movie_idx)
    
    # update user_movie 
    if user_id not in user_movie:
        user_movie[user_id] = [movie_id]
    else:
        user_movie[user_id].append(movie_id)
    
    # update movie_user 
    if movie_id not in movie_user:
        movie_user[movie_id] = [user_id]
    else:
        movie_user[movie_id].append(user_id)
    
    # update usermovie_rating 
    usermovie_rating[(user_id, movie_id)] = row.rating

df_train.apply(update_user_movie, axis = 1)
print('Updating finished')


# In[8]:



# create a dictionary to find ratings for a user and a movie for test set 
print('Update dictionary for test set')
usermovie_rating_test = {}
def update_usermovie_test(row):
    user_id = int(row.userId)
    movie_id = int(row.movieId)
    
    usermovie_rating_test[(user_id, movie_id)] = row.rating
df_test.apply(update_usermovie_test, axis = 1)

print("Updating finished")


# In[10]:


# save the dictionaries in binary format 
with open('../data/ml-20m/user_movie', 'wb') as f:
    pickle.dump(user_movie, f)
    
with open('../data/ml-20m/movie_user', 'wb') as f:
    pickle.dump(movie_user, f)
    
with open('../data/ml-20m/usermovie_rating', 'wb') as f:
    pickle.dump(usermovie_rating, f)
    
with open('../data/ml-20m/usermovie_rating_test', 'wb') as f:
    pickle.dump(usermovie_rating_test, f)

