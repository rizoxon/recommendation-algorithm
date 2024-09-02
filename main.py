import random

#### Mathematical namings
# R = utility matrix
# P = user factor matrix
# Q = item factor matrix

# Gradient descent is a mathematical technique and optimization algorithm used to find the minimum values of a function, or local minima. It's often used to train machine learning models and neural networks by minimizing errors between predicted and actual results.

ratings = [
[1, 3, 4.0],
[2, 7, 3.0],
[3, 12, 5.0],
[4, 18, 2.0],
[5, 25, 4.0],
[6, 30, 3.0],
[7, 2, 5.0],
[8, 9, 1.0],
[9, 14, 4.0],
[10, 20, 3.0],
[11, 26, 5.0],
[12, 31, 2.0],
[13, 1, 4.0],
[14, 8, 3.0],
[15, 13, 5.0],
[16, 19, 1.0],
[17, 24, 4.0],
[18, 29, 2.0],
[19, 3, 5.0],
[20, 10, 3.0],
[1, 15, 4.0],
[2, 21, 2.0],
[3, 27, 5.0],
[4, 32, 1.0],
[5, 4, 4.0],
[6, 11, 3.0],
[7, 16, 5.0],
[8, 22, 2.0],
[9, 28, 4.0],
[10, 33, 1.0],
[11, 5, 5.0],
[12, 12, 3.0],
[13, 17, 4.0],
[14, 23, 2.0],
[15, 29, 5.0],
[16, 34, 1.0],
[17, 6, 4.0],
[18, 13, 3.0],
[19, 18, 5.0],
[20, 24, 2.0],
[1, 30, 4.0],
[2, 35, 1.0],
[3, 7, 5.0],
[4, 14, 3.0],
[5, 19, 4.0],
[6, 25, 2.0],
[7, 31, 5.0],
[8, 1, 1.0],
[9, 8, 4.0],
[10, 15, 3.0],
[11, 20, 5.0],
[12, 26, 2.0],
[13, 32, 4.0],
[14, 2, 1.0],
[15, 9, 5.0],
[16, 16, 3.0],
[17, 21, 4.0],
[18, 27, 2.0],
[19, 33, 5.0],
[20, 3, 1.0],
[1, 10, 4.0],
[2, 17, 3.0],
[3, 22, 5.0],
[4, 28, 2.0],
[5, 34, 4.0],
[6, 4, 1.0],
[7, 11, 5.0],
[8, 18, 3.0],
[9, 23, 4.0],
[10, 29, 2.0],
[11, 35, 5.0],
[12, 5, 1.0],
[13, 12, 4.0],
[14, 19, 3.0],
[15, 24, 5.0],
[16, 30, 2.0],
[17, 1, 4.0],
[18, 6, 1.0],
[19, 13, 5.0],
[20, 20, 3.0],
[1, 25, 4.0],
[2, 31, 2.0],
[3, 2, 5.0],
[4, 7, 1.0],
[5, 14, 4.0],
[6, 21, 3.0],
[7, 26, 5.0],
[8, 32, 2.0],
[9, 3, 4.0],
[10, 8, 1.0],
[11, 15, 5.0],
[12, 22, 3.0],
[13, 27, 4.0],
[14, 33, 2.0],
[15, 4, 5.0],
[16, 9, 1.0],
[17, 16, 4.0],
[18, 23, 3.0],
[19, 28, 5.0],
[20, 34, 2.0],
[55, 23, 4.0]
]

##### Create Utility matrix out of an array
def create_matrix(data):
	# Extract unique users and items
	users = sorted(set(row[0] for row in data))
	items = sorted(set(row[1] for row in data))

	# Create mappings from IDs to indexes
	user_indexes = {}
	for i in range(len(users)):
		user_indexes[users[i]] = i
	
	item_indexes = {}
	for i in range(len(items)):
		item_indexes[items[i]] = i

	# Initialize the matrix with zeros
	matrix = []
	for i in range(len(users)):
		row = [0.0] * len(items)
		matrix.append(row)

	# Fill the matrix with ratings
	for user, item, rating in data:
		user_index = user_indexes[user]
		item_index = item_indexes[item]
		matrix[user_index][item_index] = rating

	return matrix, user_indexes, item_indexes, users, items
R, user_indexes, item_indexes, users, items = create_matrix(ratings)

##### Parameters
num_users = len(users)
num_items = len(items)
learning_rate = 0.01
regularization = 0.02
num_iterations = 5000

##### Initialize Factor Matrixes (user and item)
P = []
for i in range(num_users):
	P.append([random.random()])

Q = []
for i in range(num_items):
	Q.append([random.random()])

#### Generate rating by user, item matrix multiplication
def calc_rating(user, item):
	user_index = user_indexes[user]
	item_index = item_indexes[item]
	rating = P[user_index][0] * Q[item_index][0]
	return round(rating, 1)

#### Optimize Factor matrixes
for i in range(num_iterations):
	for i in range(num_users):
		for j in range(num_items):
			if R[i][j] > 0: # Only consider observed ratings
				# calculate the diffrence in prediction
				diff = R[i][j] - calc_rating(users[i], items[j])

				# Update user and item feature matrices (Gradient descent with L2 regularization)
				P[i][0] += learning_rate * (diff * Q[j][0] - regularization * P[i][0])
				Q[j][0] += learning_rate * (diff * P[i][0] - regularization * Q[j][0])

#### Final matrix with predictions
predicted_ratings = []
for i in range(num_users):
	user_ratings = []
	for j in range(num_items):
		if R[i][j] == 0: user_ratings.append(calc_rating(users[i], items[j]))
		else: user_ratings.append(R[i][j])
	predicted_ratings.append(user_ratings)


# Suggest items for a specific user
def suggest_items(user):
	recommendations = []
	for j in range(len(items)):
		if R[user_indexes[user]][j] == 0: recommendations.append((items[j], calc_rating(user, items[j])))
	recommendations.sort(key=lambda x: x[1], reverse=True)
	return recommendations

for item, rating in suggest_items(1):
	print(f"Item: {item}, Predicted Rating: {rating:.1f}")

##### Utility matrix
print("Utility Matrix:")
for row in R:
	print(row)

# print("User to Index mapping:", user_indexes)
# print("Item to Index mapping:", item_indexes)
# print("Users:", users)
# print("Items:", items)

#### Preticted ratings matrix
print("Predicted ratings Matrix:")
for r in predicted_ratings:
	print(r)
