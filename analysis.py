import json
import string
import random
from collections import defaultdict
from stop_words import get_stop_words
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce

# punctuation = set(string.punctuation)
# en_stop = get_stop_words('en')
#
# # Number of reviews
# N = 100000
# reviews = []
# stars = []
# wordCount = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
# review_len = [[],[],[],[],[]]
# rlen = []
#
# with open('data/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json', 'r',encoding='utf-8') as data_file:
#     for l in data_file:
#         if len(reviews) >= N:
#             break
#         d = json.loads(l)
#
#         stars.append(d['stars'])
#         # Ignore capitalization, punctuation and stop words
#         raw = ''.join([c for c in d['text'].lower() if not c in punctuation]).split()
#         review_len[d['stars']-1].append(len(raw))
#         rlen.append(len(raw))
#         review = []
#         for w in raw:
#             if w not in en_stop:
#                 wordCount[d['stars']-1][w] += 1
#                 review.append(w)
#         reviews.append(review)
#
#
# print("Mean review length = %f"%(reduce(lambda x,y:x+y, map(sum, review_len))/N))
# print("Mean stars = %f"%(sum(stars)/N))
#
# for i in range(5):
#     print("Mean review length for %d stars = %f"%(i+1, sum(review_len[i])/len(review_len[i])))
#
# font = {'family' : 'normal', 'size'   : 25}
# matplotlib.rc('font', **font)
# plt.plot(rlen, stars, '.')
# plt.xlabel('length of review',fontsize=25)
# plt.ylabel('stars',fontsize=25)
# plt.show()
#
# # counts = [(wordCount[w], w) for w in wordCount]
# # counts.sort()
# #
# # for i in range(len(counts)):
# #     if counts[i][0] > 1:
# #         print("Number of unique unigrams: " + str(len(wordCount)-i))
# #         break
# #
# # counts.reverse()
#
# for i in range(5):
#     counts = [(wordCount[i][w], w) for w in wordCount[i]]
#     counts.sort()
#     counts.reverse()
#     print("For %d star, the most common 10 words are:"%(i+1))
#     print(counts[:10])

MSE_train = [1.005190,0.950975,0.914319,0.881404,0.872430,0.866465,0.861473,0.856721,0.852487,0.849908,0.846085,0.843158,0.839606,0.835982,0.833529]
MSE_valid = [1.022964,0.977753,0.954110,0.925895,0.916371,0.912248,0.908868,0.906861,0.903749,0.900369,0.899170,0.897795,0.896276,0.895476,0.896774]
dim = [150,300,500,750,850,950,1050,1150,1250,1350,1450,1550,1650,1750,1850]
font = {'family' : 'normal',
        'size'   : 25}

matplotlib.rc('font', **font)
matplotlib.rc('lines',lw=4)
plt.plot(dim, MSE_train)
plt.plot(dim, MSE_valid)
plt.legend(['train','valid'])
plt.xlabel("dimension", fontsize=25)
plt.ylabel("MSE",fontsize=25)
plt.show()
