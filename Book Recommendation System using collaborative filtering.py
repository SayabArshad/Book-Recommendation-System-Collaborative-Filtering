#import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample book dataset
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'title': [
        'Book A', 'Book B', 'Book C', 'Book A', 'Book D'
        , 'Book B', 'Book C', 'Book E', 'Book A', 'Book C'],
    'rating': [5, 4, 3, 4, 5, 2, 3, 4, 5, 4]
}

# Create a DataFrame
df = pd.DataFrame(data)
print("Book Ratings DataSet:\n", df)

# Create a user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
print("\nUser Similarity Matrix:\n", user_similarity_df)

# Function to get book recommendations for a user
def recommend_books(user_id,  num_recommendations=3):
    # Get similarity scores for the given user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    
    # Get books rated by similar users
    similar_user_ids = similar_users.index[1:]  # Exclude the user themselves
    recommended_books = pd.Series(dtype=float)
    
    for similar_user in similar_user_ids:
        # Get books rated by the similar user
        similar_user_ratings = user_item_matrix.loc[similar_user]
        
        # Filter out books already rated by the target user
        unrated_books = similar_user_ratings[user_item_matrix.loc[user_id] == 0]
        
        # Add weighted ratings to the recommendations
        for book, rating in unrated_books.items():
            if book in recommended_books:
                recommended_books[book] += rating * similar_users[similar_user]
            else:
                recommended_books[book] = rating * similar_users[similar_user]
    
    # Sort and return the top N recommendations
    recommended_books = recommended_books.sort_values(ascending=False).head(num_recommendations)
    return recommended_books.index.tolist()
# Get recommendations for a specific user
user_id = 1
recommended_books = recommend_books(user_id)
print(f"\nRecommended Books for User {user_id}:", recommended_books)
# Output:
