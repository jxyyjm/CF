#!/usr/bin/python
# -*- coding:utf-8 -*-
# reference : https://github.com/ictar/python-doc/blob/master/Science%20and%20Data%20Analysis/%E5%9C%A8Python%E4%B8%AD%E5%AE%9E%E7%8E%B0%E4%BD%A0%E8%87%AA%E5%B7%B1%E7%9A%84%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F.md

# memory-based CF, user/item-based CF who will compute all-the-exits, and give recommend for the less-choice user/item.                       #
# different model-based CF, memory-based CF depend on all the infomation, and cannot deal with cold-starting problem & sparse-value problem.  #

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

### ==== 1) input data ==== ###
## Data set ##http://files.grouplens.org/datasets/movielens/m1-100k
header = ['user_id', 'item_id', 'rating', 'timestamp']
df     = pd.read_csv('../data/u.data', sep='\t', names=header)
## show the normal statistic-value ##
print df.describe()
print df.columns
## record the num-of-user and num-of-items
n_users = df['user_id'].unique().shape[0]
n_items = df['item_id'].unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items) 


### ==== 2) split train & test data ==== ###
train_data, test_data = cv.train_test_split(df, test_size = 0.25)


### ==== 3) build user-item matrix ==== ###
train_data_matrix = np.zeros((n_users, n_items))
for tup in train_data.itertuples():
	train_data_matrix[tup[1]-1, tup[2]-1] = tup[3]
## train_data_matrix 包括了除了测试集之外的所有评分:: 主要用来保存 训练用户对所有item的评分 ##
test_data_matrix  = np.zeros((n_users, n_items))
for tup in test_data.itertuples():
	test_data_matrix[tup[1]-1, tup[2]-1] = tup[3]
## test_data_matrix 包括了除了训练集之外的所有评分:: 主要用来保存测试集数据 ##

# ? why use two user-item matrix   ? # because one will train and give the predicted test-set ## the other named test is used to store the test set #
# ? why shape is n_users * n_items ? # because train-matrix will give the predicted test-set-part, and same-struct is easy to evalute #
#									 # here, user-based or item-based is compute *all-similarity*, so dim need all-dim #
#									 # above, why need compute all-similarity, because for predicting sth-user/item, complete-info will be easy to do some-process later #
#									 # also, not compute all-similarity, later when predict, it will compute single for who-will-be-predicted #
#									 # also, a potential problem is that, real-word, all-user/item is has less-choice than no-choice. it will be odd if not compute all-similarity #
#									 # also, I find that in real-word, user/item-based CF is good at to predict all-has-exits, it does bad work for one-never-exits #

### ==== 4) compute distance among ==== ###
user_similarity = pairwise_distances(train_data_matrix, metric='cosine') ## 各个行之间的相关性 ##
item_similarity = pairwise_distances(train_data_matrix.T,  metric='cosine')
print type(user_similarity), user_similarity.shape, 'user_similarity' ## n_users * n_users ##
print type(item_similarity), item_similarity.shape, 'item_similarity' ## n_items * n_items ##
print user_similarity[0:20]
print item_similarity[0:20]


### ==== 5) predict by diff type ==== ###
def predict(ratings, similarity, type='user'):
	if type == 'user':
		mean_user_rating = ratings.mean(axis=1) ## 获取打分的均值
		ratings_diff     = (ratings - mean_user_rating[:, np.newaxis]) ## np.newaxis 是用来规范维度的，可以将(6,)->(6,1)
		## 打分与均值之间的差距 ##
		pred             = mean_user_rating[:, np.newaxis] \
						   + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
	elif type == 'item':
		pred             = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
	return pred
user_prediction = predict(train_data_matrix, user_similarity, type='user')
item_prediction = predict(train_data_matrix, item_similarity, type='item')
print '========= after predict ========='
print type(user_prediction), user_prediction.shape, 'user_prediction'
print type(item_prediction), item_prediction.shape, 'item_prediction'
print user_prediction[0:10]
print item_prediction[0:10]

### ==== 6) evaluate the result ==== ###
#RMSE = sqrt[(sum(xi-x_mean)^2)/N]
def rmse(prediction, ground_truth):
	prediction   = prediction[ground_truth.nonzero()].flatten()
	ground_truth = ground_truth[ground_truth.nonzero()].flatten()
	return sqrt(mean_squared_error(prediction, ground_truth))
print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))








