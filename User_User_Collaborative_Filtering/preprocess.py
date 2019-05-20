
import pandas as pd

# https://www.kaggle.com/grouplens/movielens-20m-dataset
# https://grouplens.org/datasets/movielens/20m/
df = pd.read_csv('../data/ml-20m/rating.csv')


# note:
# user ids are ordered sequentially from 1..138493 with no missing numbers
# movie ids are integers from 1..131262
# NOT all movie ids appear
# there are only 26744 movie ids
# write code to check it yourself!


# make the user ids go from 0...N-1
df.userId = df.userId - 1

# create a mapping for movie ids so ids are consecutive 
unique_movie_ids = set(df.movieId.values)
movie_to_idx = {}
count = 0
for movie_id in unique_movie_ids:
    movie_to_idx[movie_id] = count
    count += 1

# add mapped id back to data 
df['movie_idx'] = df.apply(lambda row: movie_to_idx[row.movieId], axis = 1)

df = df.drop(columns=['timestamp'])

df.to_csv('../data/ml-20m/edited_rating.csv', index = False)