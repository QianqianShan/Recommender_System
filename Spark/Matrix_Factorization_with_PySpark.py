#!/usr/bin/env python
# coding: utf-8

# install java, scala, spark and modify SPARK_HOME according to 
# https://medium.com/devilsadvocatediwakar/installing-apache-spark-on-ubuntu-8796bfdd0861
# 
# pip3 install pyspark
# 
# **This code is run by `pyspark Matrix_Factorization_with_Spark.py` locally in command line on Ubuntu 18.04**

# In[ ]:


import os
# import alternatingly least squares, Rating object 
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# load data 
data = sc.textFile('~/Desktop/Repos/Big_Data_Projects/Recommender_System/data/ml-20m/small_ratings.csv')

# the first row is header
header = data.first()
# skip header row 
data = data.filter(lambda row: row != header)

# convert data to a sequence of Rating objects 
# (a triple of userId, movieId, rating)
ratings = data.map(lambda row:row.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# split data into train and test sets 
train, test = ratings.randomSplit([0.8, 0.2])

# build model 
K = 10 # latent dimensionality 
epochs = 15 
model = ALS.train(train, K, epochs)

# evaluate the model 
# x: (userId, movieId)
x = train.map(lambda p:(p[0], p[1]))
# make predictions, p is a tuple of (userId, movieId) and (predicted rating) 
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))

# join actual and predicted ratings based on (userId, movieId)
actual_and_preds = train.map(lambda r:((r[0], r[1]), r[2])).join(p)
# actual_and_preds has format of ((userId, movieId), (actual rating, predicted rating))

# look at the first 5 rows 
actual_and_preds.take(5)

# mse 
mse = actual_and_preds.map(lambda r:(r[1][0] - r[1][1])**2).mean()
print('train mse is %s' %mse)


# similarly for test mse 
x = test.map(lambda p: (p[0], p[1]))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = test.map(lambda r: ((r[0], r[1]), r[2])).join(p)
mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("test mse: %s" % mse)

