import json

# Number of reviews
N = 100000
stars = []

with open('data/yelp_dataset_challenge_round9/yelp_academic_dataset_review.json', 'r',encoding='utf-8') as data_file:
    for l in data_file:
        if len(stars) >= N:
            break
        d = json.loads(l)
        stars.append(d['stars'])

Y_train = stars[:90000]
Y_valid = stars[90000:]
pred = sum(Y_train) / len(Y_train)
print(pred)

# Test on training set
MSE = 0
for y in Y_train:
    MSE += (y - pred) **2
MSE /= len(Y_train)
print("Training MSE = %f"%(MSE))

# Test on validation set
MSE = 0
for y in Y_valid:
    MSE += (y - pred) **2
MSE /= len(Y_valid)
print("Validation MSE = %f"%(MSE))
