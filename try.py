import json
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models
import gensim

N = 10
i = 0
reviews = []
with open('data/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json', 'r',encoding='utf-8') as data_file:
    for l in data_file:
        if i > N:
            break
        d = json.loads(l)
        print(d)
        i += 1

# categCount = defaultdict(int)
# with open('data/yelp_dataset_challenge_round9/yelp_academic_dataset_business.json', 'r',encoding='utf-8') as data_file:
#     for l in data_file:
#         d = json.loads(l)
#         if d['categories'] is None:
#             continue
#         for c in d['categories']:
#             categCount[c] += 1
#
# counts = [(categCount[c], c) for c in categCount]
# counts.sort()
# counts.reverse()
# for c in counts[:15]:
#     print(c)

# reviews = []
# with open('data/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json', 'r',encoding='utf-8') as data_file:
#     for l in data_file:
#         d = json.loads(l)
#         if d['business_id'] not in idSet:
#             continue
#         reviews.append(d)
#
# print("Number of reviews: %d"%(len(reviews)))
#
# tokenizer = RegexpTokenizer(r'\w+')
#
# # create English stop words list
# en_stop = get_stop_words('en')
#
# # Create p_stemmer of class PorterStemmer
# p_stemmer = PorterStemmer()
#
# texts = []
# for review in reviews:
#     # clean and tokenize document string
#     raw = review['text'].lower()
#     tokens = tokenizer.tokenize(raw)
#
#     # remove stop words from tokens
#     stopped_tokens = [i for i in tokens if not i in en_stop]
#
#     # stem tokens
#     stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
#
#     # add tokens to list
#     texts.append(stemmed_tokens)
#
# # turn our tokenized documents into a id <-> term dictionary
# dictionary = corpora.Dictionary(texts)
#
# # convert tokenized documents into a document-term matrix
# corpus = [dictionary.doc2bow(text) for text in texts]
#
# # generate LDA model
# ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=8, id2word = dictionary, passes=100)
#
# print(ldamodel.print_topics())
