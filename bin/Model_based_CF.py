#!/usr/bin/python
# -*- coding:utf-8 -*-

# reference : http://www.jscon.co/multiarray/rs_used_svd.html ## not all right #
# reference : A Guide to Singular Value Decomposition for Collaborative Filtering
# reference : Netflix Prize and SVD
# reference : Item-Based Collaborative Filtering Recommendation Algorithms
# reference : Applying SVD on Generalized Item-based Filtering

# model-based CF, it will try to deal with the cold-starting & sparse-value problem #
# There are many model-based CF algorithms:
#		Bayesian networks
#		clustering models
#		latent semantic models such as singular value decomposition,
#								   probabilistic latent semantic analysis,
#								   multiple multiplicative factor,
#								   latent Dirichlet allocation,
#								   Markov decision process based models.

# here we use SVD to reach a simple model-based CF

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


### ==== 4) sparsity evaluate here  ==== ####
sparsity = round(1.0 -len(df)/float(n_users*n_items), 3) ## 1 - 矩阵中有值的数据个数/矩阵的大小 ##
print 'The sparsity level of Movielens100k is ' + str(sparsity*100)


### ==== 5) svd ==== ###
# X = USV^
import scipy.sparse as sp
from scipy.sparse.linalg import svds

u, s, vt = svds(train_data_matrix, k=20)
s_diag_matrix = np.diag(s)
x_pred   = np.dot(np.dot(u, s_diag_matrix), vt) ## 提取了top-20奇异值特征维度之后的 train_data_matrix的表示 ##
print str(u.shape), '\tu.shape\t\t\t', type(u)
print str(s.shape), '\t\ts.shape\t\t\t', type(s)
print str(s_diag_matrix.shape), '\ts_diag_matrix.shape\t', type(s_diag_matrix)
print str(vt.shape), '\tvt.shape\t\t', type(vt)
print str(x_pred.shape), '\tx_pred.shape\t\t', type(x_pred)
print str(train_data_matrix.shape), '\ttrain_data_matrix\t', type(train_data_matrix)
print x_pred[0:2, 0:10]
print train_data_matrix[0:2, 0:10]


#下面这句很奇怪，SVD是分解，投射到某特征空间上的表示 #
# 应该是用x_pred这个也是原来train_data_matrix的新特征空间的表示 #
## 1.2 如果对未知的用户 推荐 ##
## X_new * vt * s^(-1) = U_new ## 将该用户的向量映射到u空间上去。这个说法是有问题的 #
## 1.3 如果是对新的item 
## X_new * u * s^(-1) = I_new ## 将该item的向量映射到v空间上去。 这个说法是有问题的 #

# with my understanding, the correct way is as following 
# u*sqrt(s) as the user-feature-array
# sqrt(s)*vt as the item-feature-array
# user_similarity = pairwise_distances(u*sqrt(s), metric='cosine')
# item_similarity = pairwise_distances(sqrt(s)*vt, metric='cosine')
# if one-never-exit user/item wanted tobe predicted 
#    A = USV^ ====> u' = new * V^(-1) * S^(-1)
#    then compute the similarity between u' among other users
#    as same as item

user_train_matrix = u.dot(np.sqrt(s_diag_matrix))
item_train_matrix = np.sqrt(s_diag_matrix).dot(vt)
print str(user_train_matrix.shape), '\tuser_train_matrix\t', type(user_train_matrix)
print str(item_train_matrix.shape), '\titem_train_matrix\t', type(item_train_matrix)
user_similarity = pairwise_distances(user_train_matrix, metric='cosine')
item_similarity = pairwise_distances(item_train_matrix.T, metric='cosine')
print str(user_similarity.shape), '\tuser_similarity\t\t', type(user_similarity)
print str(item_similarity.shape), '\titem_similarity\t\t', type(item_similarity)


### ==== 6) predict by diff type ==== ###
def rmse(prediction, ground_truth):
	prediction   = prediction[ground_truth.nonzero()].flatten()
	ground_truth = ground_truth[ground_truth.nonzero()].flatten()
	return sqrt(mean_squared_error(prediction, ground_truth))

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
print user_prediction.shape, '\tuser_prediction\t\t', type(user_prediction)
print item_prediction.shape, '\titem_prediction\t\t', type(item_prediction)
print user_prediction[0:10]
print item_prediction[0:10]
print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))

