import json
import string
import math
import numpy as np
from collections import defaultdict
from stop_words import get_stop_words
from sklearn import linear_model
import scipy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create list of punctuation and stop words
punctuation = set(string.punctuation)
en_stop = get_stop_words('en')

# Number of reviews
N = 100000
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
words = [x[1] for x in counts[:2000]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

# # Transform the words into bag of words
# bag_of_words = []
# for review in reviews:
#     bow = [0.0] * len(words)
#     for word in review:
#         if word in wordSet:
#             bow[wordId[word]] += 1
#     bag_of_words.append(bow)

# Calculate idf for all words
idf = defaultdict(int)
for review in reviews:
    for word in set(review):
        if word in wordSet:
            idf[word] += 1
for word in words:
    idf[word] = math.log(N/idf[word])

# Calculate tf-idf for all words
tf_idfs = []
for review in reviews:
    tf_idf =[0] * len(words)
    for word in review:
        if word in wordSet:
            tf_idf[wordId[word]] += idf[word]
    tf_idfs.append(tf_idf)


# Do SVD on the term-document matrix
dim = 1000
data_mat = scipy.sparse.csr_matrix(tf_idfs)
u, sigma, vt = scipy.sparse.linalg.svds(data_mat,k=dim)

X = []
s = np.linalg.inv(scipy.linalg.diagsvd(sigma,dim,dim))
v = np.transpose(vt)
for d in data_mat:
    a = normalize(np.dot(d.dot(v), s), norm='l1') * 100
    a = a.reshape(dim).tolist()
    a.append(1)
    X.append(a)

X_train = X[:90000]
Y_train = stars[:90000]
X_valid = X[90000:]
Y_valid = stars[90000:]

clusters = 2
kstars = [0]*clusters
klen = [0]*clusters
kmeans = KMeans(n_clusters=clusters).fit(X_train)
for i in range(90000):
    kstars[kmeans.labels_[i]] += Y_train[i]
    klen[kmeans.labels_[i]] += 1
for i in range(clusters):
    print(kstars[i]/klen[i])


# # Linear regression with regularizer
# clf = linear_model.Ridge(1.0, fit_intercept=False)
# clf.fit(X_train, Y_train)
# theta = clf.coef_
#
# # Test on training set
# predictions = clf.predict(X_train)
# MSE = 0
# for z in zip(predictions, Y_train):
#     MSE += (z[0] - z[1]) **2
# MSE /= len(Y_train)
# print("Training MSE = %f"%(MSE))
#
# # Test on validation set
# predictions = clf.predict(X_valid)
# MSE = 0
# for z in zip(predictions, Y_valid):
#     MSE += (z[0] - z[1]) **2
# MSE /= len(Y_valid)
# print("Validation MSE = %f"%(MSE))
