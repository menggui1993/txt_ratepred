import json
import string
import random
from collections import defaultdict
from stop_words import get_stop_words
from sklearn import linear_model
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

# Transform the words into bag of words
bag_of_words = []
for review in reviews:
    bow = [0] * len(words)
    for word in review:
        if word in wordSet:
            bow[wordId[word]] += 1
    bow.append(1)
    bag_of_words.append(bow)

X_train = bag_of_words[:90000]
Y_train = stars[:90000]
X_valid = bag_of_words[90000:]
Y_valid = stars[90000:]

# Linear regression with regularizer
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X_train, Y_train)
theta = clf.coef_

# Test on training set
predictions = clf.predict(X_train)
MSE = 0
for z in zip(predictions, Y_train):
    MSE += (z[0] - z[1]) **2
MSE /= len(Y_train)
print("Training MSE = %f"%(MSE))

# Test on validation set
predictions = clf.predict(X_valid)
MSE = 0
for z in zip(predictions, Y_valid):
    MSE += (z[0] - z[1]) **2
MSE /= len(Y_valid)
print("Validation MSE = %f"%(MSE))

# Plot the word cloud for 50 most positive and most negative words
thw = list(zip(words, theta))
thw.sort(key=lambda x: x[1])
print("10 most negative words:")
for z in thw[:10]:
    print(z)
print("10 most positive words:")
for z in thw[-10:]:
    print(z)

pos_words = dict(thw[-50:])
neg_words = dict(map(lambda x: (x[0], -x[1]), thw[:50]))

def red_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(%d, 100%%, 50%%)" % random.randint(0, 40)

def blue_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(%d, 100%%, 50%%)" % random.randint(180, 220)

posWC = WordCloud(background_color='white',width=1200,height=600).generate_from_frequencies(pos_words)
plt.imshow(posWC.recolor(color_func=red_func,random_state=3))
plt.axis("off")

plt.figure()
negWC = WordCloud(background_color='white',width=1200,height=600).generate_from_frequencies(neg_words)
plt.imshow(negWC.recolor(color_func=blue_func,random_state=3))
plt.axis("off")
plt.show()
