import json
import string
import random
import numpy as np
from collections import defaultdict
from stop_words import get_stop_words
from sklearn import linear_model
import scipy
from sklearn.preprocessing import normalize

# Create list of punctuation and stop words
punctuation = set(string.punctuation)
en_stop = get_stop_words('en')

# Number of reviews
N = 60000
reviews = []
stars = []
wordCount = defaultdict(int)

with open('data/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json', 'r',encoding='utf-8') as data_file:
    for l in data_file:
        if len(reviews) >= N:
            break
        d = json.loads(l)

        stars.append(d['stars'])
        # Ignore capitalization, punctuation and stop words
        raw = ''.join([c for c in d['text'].lower() if not c in punctuation])
        review = []
        for w in raw.split():
            if w not in en_stop:
                wordCount[w] += 1
                review.append(w)
        reviews.append(review)

# Use the 1000 most common words
counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()
words = [x[1] for x in counts[:1000]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

# Transform the words into bag of words
bag_of_words = []
for review in reviews:
    bow = [0.0] * len(words)
    for word in review:
        if word in wordSet:
            bow[wordId[word]] += 1
    bow.append(1)
    bag_of_words.append(bow)

# Do SVD on the term-document matrix
dim = 500
data_mat = scipy.sparse.csr_matrix(bag_of_words)
u, sigma, vt = scipy.sparse.linalg.svds(data_mat,k=dim)

X = []
s = np.linalg.inv(scipy.linalg.diagsvd(sigma,dim,dim))
v = np.transpose(vt)
for d in data_mat:
    a = normalize(np.dot(d.dot(v), s), norm='l1') * 100
    a = a.reshape(dim).tolist()
    a.append(1)
    X.append(a)

X_train = X[:50000]
Y_train = stars[:50000]
X_valid = X[50000:]
Y_valid = stars[50000:]

# Linear regression with regularizer
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X_train, Y_train)
theta = clf.coef_

# Test on validation set
predictions = clf.predict(X_valid)
MSE = 0
for z in zip(predictions, Y_valid):
    MSE += (z[0] - z[1]) **2
MSE /= len(Y_valid)
print("MSE = %f"%(MSE))
