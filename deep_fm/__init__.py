import time
import sys
from pylab import *
from scipy import sparse
import numpy as np

import tensorflow as tf

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

         observed_features_validation,
              labels_validation,
              rank,
              max_iter=100,
              verbose=False,
              lambda_v=0,
              lambda_k=0,
              lambda_w=0,
              lambda_constants=0,
              epsilon=0.001,
              optimizer=tf.train.AdamOptimizer(),
              depth=3,
              seed=12345):

    # Extract info about shapes etc from the training data
    num_items = observed_features.shape[0]
    num_features = observed_features.shape[1]
    
    # matrix defining the inner product weights when doing interactions
    K = tf.Variable(tf.truncated_normal([rank, rank], stddev=0.2, mean=0, seed=seed), name="metric_matrix")
    
    # coefficients for linear function on inputs (wide part)
    w = tf.Variable(tf.truncated_normal([1, num_features], stddev=0.2, mean=0, seed=seed), name="hyperplane")

    # coefficients for linear functinos on inputs (deep part)
    lw = tf.Variable(tf.truncated_normal([1, rank], stddev=0.2, mean=0, seed=seed), name="latenthyperplane")

    # bias in linear function
    b = tf.Variable(tf.truncated_normal([1, 1], stddev=0.2, mean=0, seed=seed), name="b_one")
    
    x = tf.placeholder(tf.float32, [None, num_features])
    y = tf.placeholder(tf.float32)
    
    norm_x = tf.nn.l2_normalize(x, dim=0)
    
    Vx = make_embeddings(tf.transpose(norm_x), rank, num_features, depth=depth, seed=seed)
    right_kern = tf.matmul(K, Vx)
    
    full_kern = tf.matmul(tf.transpose(Vx), right_kern)
    linear = tf.matmul(w, tf.transpose(norm_x))
    latent_linear = tf.matmul(lw, Vx)

    pred = tf.reduce_sum(tf.sigmoid(linear + latent_linear + full_kern + b))
    
    # todo: dropout. currently no regularization on the interaction layers in the cost functino
    # can handle with FTRL optimization
    cost = tf.reduce_mean(-y*tf.log(pred + 0.0000000001) - (1-y)*tf.log((1-pred + 0.0000000001)) + 
            lambda_k*tf.nn.l2_loss(K) +
            lambda_w*tf.nn.l2_loss(w) +
            lambda_constants*tf.nn.l2_loss(b))
    optimize = optimizer.minimize(cost)
    norm = tf.reduce_mean(tf.nn.l2_loss(w))
    
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)
        last_cost = 1000000
        for iter in range(0, max_iter):
            avg_cost = 0
            
            for i in range(num_items):
                _, c, n = sess.run([optimize, cost, norm],
                              feed_dict={x:observed_features[i].reshape(1, num_features), y:labels[i]})
                avg_cost += c / num_items
            if verbose:
                print("epoch: %s, cost: %s" % (iter+1, avg_cost))

            # check for convergence
            if abs(avg_cost-last_cost)/avg_cost < epsilon:
                break
                
            last_cost = avg_cost
            
        if verbose:
            print("optimization finished")
        predictions = []
        total_costs = 0
        for i in range(observed_features_validation.shape[0]):
            p, c = sess.run([pred, cost], feed_dict={x:observed_features_validation[i].reshape(1, num_features), y:labels_validation[i]})
            predictions.append(p)
            total_costs += c
        return predictions, total_costs/observed_features_validation.shape[0], sess.run([norm])


def test_fm():

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

    ng = datasets.fetch_20newsgroups (categories=categories, shuffle=True)
    labels = [1 if y == 2 else 0 for y in ng.target.reshape(-1,1)]

    tfidf = TfidfVectorizer(decode_error=False, min_df=5)

    X_train, X_test, y_train, y_test = train_test_split(ng.data, labels, test_size=.3)
    X_train = tfidf.fit_transform(X_train).todense()
    X_test = tfidf.transform(X_test).todense()

    r = 10
    predictions, test_costs, norm = factorize(X_train, y_train, X_test, y_test, r, verbose=True, lambda_v=0.1, max_iter=300)
    print("rank: %s, cost: %s, overall AUC: %s, norm: %s") % (r, test_costs, roc_auc_score(y_test, predictions, average="weighted"), norm)
