#!/usr/bin/env python
# coding: utf-8

# Use the preprocessed data to apply user-user CF and make predictions.

# In[31]:


import os 
import pickle 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 
from datetime import datetime 
from sortedcontainers import SortedList 


# In[32]:


# load data 
os.chdir(r'/home/qshan/Desktop/Repos/Big_Data_Projects/Recommender_System/User_User_Collaborative_Filtering')
with open('../data/ml-20m/user_movie', 'rb') as f:
    user_movie = pickle.load(f)
 
with open('../data/ml-20m/movie_user', 'rb') as f:
    movie_user = pickle.load(f)
    
with open('../data/ml-20m/usermovie_rating', 'rb') as f:
    usermovie_rating = pickle.load(f)
    
with open('../data/ml-20m/usermoive_rating_test', 'rb') as f:
    usermovie_rating_test = pickle.load(f)


# In[34]:


#  number of users and movies 
N = np.max(list(user_movie.keys())) + 1 

M = max(np.max(list(movie_user.keys())), np.max([m_id for (_, m_id), _ in usermovie_rating_test.items()])) + 1

print('Max of users id, N =', N, '\n')
print('Max movies id, M =', M, '\n')


# In[35]:


# number of neighbors to be used,drop the users with low weight w_ij 
K = 25 

# number of common movies users must have in order to compute correlation among users 
limit = 5 

# list to save lists of neighbor info (-w_ij, j) for each user 
neighbors = []

# users' average ratings 
averages = []

# users' deviations 
deviations = []

# loop through each user for find K closest neighbors to user i 
for i in range(N):
    # movies rated by user i 
    movies_i = set(user_movie[i])
    # corresponding ratings of user i of above movies 
    ratings_i = {movie:usermovie_rating[(i, movie)] for movie in movies_i}
    # average rating 
    avg_i = np.mean(list(ratings_i.values()))
    
    # r_ij - bar(r)_i
    dev_i = {movie: (rating - avg_i) for movie, rating in ratings_i.items()}
    
    dev_i_vals = np.array(list(dev_i.values()))
    
    # sigma used for denominator in w_ij
    sigma_i = np.sqrt(np.dot(dev_i_vals, dev_i_vals))
    
    # append avg_i list to averages 
    averages.append(avg_i)
    
    # append dev_i to deviations
    deviations.append(dev_i)
    
    # sorted list to sort weights for user i's neighbors 
    sl = SortedList()
    # loop through all other users and compute their weights 
    for j in range(N):
        # user j is user i, skip 
        if j == i:
            continue 
        
        movies_j = set(user_movie[j])
        
        # intersection 
        common_movies = movies_i & movies_j 
        
        # if user i and user j don't have enough rated movies in common, don't compute correlations 
        if len(common_movies) <= limit:
            continue 
            
        ratings_j = {movie:usermovie_rating[(j, movie)] for movie in movies_j}
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = {movie:(rating - avg_j) for movie, rating in ratings_j.items()}
        dev_j_vals = np.array(list(dev_j.values()))
            
        sigma_j = np.sqrt(np.dot(dev_j_vals, dev_j_vals))
        
        # compute weight 
        w_ij = sum(dev_i[m] * dev_j[m] for m in common_movies) / (sigma_i * sigma_j)
        
        # use -w as sorted list is sorted in ascending order 
        sl.add((-w_ij, j))
        
        # only keep top K highest correlations
        if len(sl) > K:
            del sl[-1]
    # store neighbor info of user i         
    neighbors.append(sl)
    if i % 200 == 0:
        print(i)


# In[ ]:


# make prediction for user i and a movie m
def predict(i, m):
    
    numerator, denominator = 0, 0
    for negative_w_ij, j in neighbors[i]:
        # user j not necessarily rated movie 
        try:
            numerator += -negative_w_ij * deviations[j][m]
            denominator += abs(negative_w_ij)
        except KeyError:
            pass 
    # make sure denominator is not zero 
    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]
    # convert unbounded prediction to be withing [0.5, 5]
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

for (i, m), target in usermovie_rating.items():
    prediction = predict(i, m)
    train_predictions.append(prediction)
    train_targets.append(target)

# prediction on test set 
test_predictions = []
test_targets = []

for (i, m), target in usermovie_rating_test.items():
    prediction = predict(i, m)
    test_predictions.append(prediction)
    test_targets.append(target)
    
print('Train RMSE:', rmse(train_predictions, train_targets))
print('Test RMSE:', rmse(test_predictions, test_targets))
    

