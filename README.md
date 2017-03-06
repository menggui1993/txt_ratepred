# Rating prediction with review text

Final project for CSE 258: Recommender system and web mining.

Dataset: 60000 reviews in yelp_dataset_challenge_round9. train:validation = 8:2. 

Model: Linear regression with regularizer

## Bag of Words

Use the 1000 most common words, build the BoW feature for every document.

## TF-IDF

Use the 1000 most common words, build the tf-idf feature for every document.

## Latent Semantic Analysis

Use the 1000 most common words, build the BoW feature. Run SVD on the term-document matrix and lower the dimension of the matrix.
