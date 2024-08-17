# import pandas as pd

# movies = pd.read_csv("movies.csv")
# ratings = pd.read_csv("ratings.csv")



### Bayesian Average
# When an item has only a few ratings, a simple average can be unreliable. For instance, an item with one 5-star rating out of 5 might seem better than an item with an average of 4.5 stars from 1000 ratings. The Bayesian Average helps mitigate this issue.
# BA = (C * M + Sum(Ratings)) / (C + N)
# BA is the Bayesian Average
# C is a constant (often the average number of ratings across all items)
# M is the mean rating across all items
# Sum(Ratings) is the sum of the ratings for the specific item
# N is the number of ratings for the specific item

# movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
# movie_stats.columns = movie_stats.columns.droplevel()

# C = movie_stats['count'].mean()
# m = movie_stats["mean"].mean()
#
# def bayesian_avg(ratings):
#	bayesian_avg = (C*m+ratings.sum())/(C+ratings.count())
#	return bayesian_avg
#
# bayesian_avg_ratings = ratings.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
# bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
# movie_stats = movie_stats.merge(bayesian_avg_ratings, on='movieId')
# movie_stats = movie_stats.merge(movies[['movieId', 'title']])


### Creating the matrix
# matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
# matrix = matrix.fillna(0)
#
# ### Sparcity -> the ratio of the number of zero-valued elements to the total number of elements in the matrix
# def calc_sparcity(matrix):
#	total_elements = matrix.size
#	zero_elements = (matrix == 0).sum().sum()
#	sparsity = zero_elements / total_elements
#	return sparsity
# sparsity = calc_sparcity(matrix)

ratings = [
	[1,1,4],
	[1,3,4],
	[1,6,4],
	[1,47,5],
	[2,50,5],
	[2,1,4],
	[2,3,4],
	[2,6,4],
	[3,47,5],
	[3,50,5],
	[3,1,4],
	[4,3,4],
	[4,6,4],
	[5,47,5],
	[5,50,5],
	[6,1,4],
	[7,3,4],
	[8,6,4],
	[9,47,5],
	[9,50,5]
]

def create_matrix(data):
	# Extract unique users and items
	users = sorted(set(row[0] for row in data))
	items = sorted(set(row[1] for row in data))

	# Create mappings from IDs to indices
	# user_to_index = {user: i for i, user in enumerate(users)}
	# item_to_index = {item: i for i, item in enumerate(items)}
	user_to_index = {}
	for i in range(len(users)):
		user_to_index[users[i]] = i
	
	item_to_index = {}
	for i in range(len(items)):
		item_to_index[items[i]] = i

	# Initialize the matrix with zeros
	# matrix = [[0.0 for _ in items] for _ in users]
	matrix = []
	for i in range(len(users)):
		row = [0] * len(items)
		matrix.append(row)

	# Fill the matrix with ratings
	# for user, item, rating in data:
	#	user_index = user_to_index[user]
	#	item_index = item_to_index[item]
	#	matrix[user_index][item_index] = rating

	return matrix, user_to_index, item_to_index, users, items


matrix, user_to_index, item_to_index, users, items = create_matrix(ratings)

# Print the matrix
print("User-Item Matrix:")
for row in matrix:
	print(row)

print("\nUser to Index mapping:", user_to_index)
print("Item to Index mapping:", item_to_index)
print("Users:", users)
print("Items:", items)
