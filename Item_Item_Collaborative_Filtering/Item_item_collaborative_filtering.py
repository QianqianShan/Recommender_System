#!/usr/bin/env python
# coding: utf-8

# Item-item CF uses the same preprocessed data as user-user CF, see how the data are preprocess from scripts in **User_User_Collaborative_Filtering folder**. 

# In[1]:


import os 
import pickle 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 
from datetime import datetime 
from sortedcontainers import SortedList 


# In[2]:


# load data 
os.chdir(r'/home/qshan/Desktop/Repos/Big_Data_Projects/Recommender_System/User_User_Collaborative_Filtering')
with open('../data/ml-20m/user_movie', 'rb') as f:
    user_movie = pickle.load(f)
 
with open('../data/ml-20m/movie_user', 'rb') as f:
    movie_user = pickle.load(f)
    
with open('../data/ml-20m/usermovie_rating', 'rb') as f:
    usermovie_rating = pickle.load(f)
    
with open('../data/ml-20m/usermovie_rating_test', 'rb') as f:
    usermovie_rating_test = pickle.load(f)


# In[3]:


#  number of users and movies 
N = np.max(list(user_movie.keys())) + 1 

M = max(np.max(list(movie_user.keys())), np.max([m_id for (_, m_id), _ in usermovie_rating_test.items()])) + 1

print('Max of users id, N =', N, '\n')
print('Max movies id M, =', M, '\n')


# In[ ]:


# number of neighbors to be used,drop the items with low weight w_ij 
K = 25 

# number of common users must have for each item in order to compute correlation among items 
limit = 5 

# list to save lists of neighbor info (-w_ij, j) for each item 
neighbors = []

# items' average ratings 
averages = []

# items' deviations 
deviations = []


# loop through each movie to find K cloest neighbors to item i 
for i in range(M):
    # users rated movie i 
    users_i = set(movie_user[i])
    
    # average and deviation 
    # ratings for movie i 
    ratings_i = {user:usermovie_rating[(user, i)] for user in users_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {user: rating - avg_i for user, rating in ratings_i.items()}
    dev_i_vals = np.array(list(dev_i.values()))
    
    sigma_i = np.sqrt(np.dot(dev_i_vals, dev_i_vals))
    
    averages.append(avg_i)
    deviations.append(dev_i)
    
    sl = SortedList()
    for j in range(M):
        if j == i:
            continue 
        users_j = set(movie_user[j])
        common_users = users_i & users_j
        
        if len(common_users) <= limit:
            continue 
        ratings_j = {user:usermovie_rating[(user, j)] for user in users_j}
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = {user:rating - avg_j for user, rating in ratings_j.items()}
        dev_j_vals = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(np.dot(dev_j_vals, dev_j_vals))
        
        # corrrelations 
        w_ij = sum(dev_i[u] * dev_j[u] for u in common_users) / (sigma_i * sigma_j)
        
        sl.add((-w_ij, j))
        if len(sl) > K:
            del sl[-1]
    neighbors.append(sl)
    if i % 200 == 1:
        print(i)


# In[ ]:


# prediction 
def predict(i, u):
    numerator, denominator = 0, 0
    for negative_w_ij, j in neighbors[i]:
        try:
            numerator += -negative_w_ij * deviations[j][u]
            denominator += abs(negative_w_ij)
        except KeyError:
            pass
    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]
    
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    return prediction 

# RMSE 
def rmse(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    return np.sqrt(np.mean((y - y_hat)**2))


# In[ ]:


# predicted score for each train example 
train_predictions = []

# the true score 
train_targets = []

for (u, m), target in usermovie_rating.items():
    prediction = predict(m, u)
    train_predictions.append(prediction)
    train_targets.append(target)

# prediction on test set 
test_predictions = []
test_targets = []

for (u, m), target in usermovie_rating_test.items():
    prediction = predict(m, u)
    test_predictions.append(prediction)
    test_targets.append(target)
    
print('Train RMSE:', rmse(train_predictions, train_targets))
print('Test RMSE:', rmse(test_predictions, test_targets))

