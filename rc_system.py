# Importing Libraries

import pandas as pd 
from surprise import SVD
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the dataset
df = pd.read_csv("amazon_ratings_electronics.csv")
df.columns


# Preparing data for Surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['User_ID', 'Product_ID', 'Rating']], reader)

# Splitting the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Building the recommendation model using SVD
algo = SVD()
algo.fit(trainset)

# Making predictions and evaluating the model
predictions = algo.test(testset)
print("\nModel Performance:")
print(f"RMSE: {accuracy.rmse(predictions)}")

# code for just one value using index

# # Generate Recommendations
# def get_top_recommendations(user_id, df, algo, top_n=10):
#     unique_products = df['Product_ID'].unique()
#     recommendations = []
#     for product_id in unique_products:
#         prediction = algo.predict(user_id, product_id)
#         recommendations.append((product_id, prediction.est))
#     recommendations.sort(key=lambda x: x[1], reverse=True)
#     return recommendations[:top_n]

# # Example: Generate recommendations for a specific user
# example_user = df['User_ID'].iloc[5]  
# recommendations = get_top_recommendations(example_user, df, algo)

# print(f"\nTop Recommendations for User {example_user}:")
# for idx, (product_id, predicted_rating) in enumerate(recommendations, 1):
#     print(f"{idx}. Product ID: {product_id} | Predicted Rating: {predicted_rating:.2f}")


# *********************************__________________________****************************************



# code for getting multiple indexes at a time

# Function to get top recommendations for a single user
def get_top_recommendations(user_id, df, algo, top_n=10):
    unique_products = df['Product_ID'].unique()
    recommendations = []
    for product_id in unique_products:
        prediction = algo.predict(user_id, product_id)
        recommendations.append((product_id, prediction.est))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

# Select 5 users from the dataset
num_users = 5
unique_users = df['User_ID'].unique()[:num_users]  

# Loop through each user and generate recommendations
for user_idx, example_user in enumerate(unique_users, start=1):
    print(f"\nTop Recommendations for User {example_user}:")
    
    recommendations = get_top_recommendations(example_user, df, algo)

    # Print recommendations in batches of 5 per user
    batch_size = 5
    for i in range(0, len(recommendations), batch_size):
        batch = recommendations[i : i + batch_size]  # Get 5 recommendations per batch
        for idx, (product_id, predicted_rating) in enumerate(batch, start=i + 1):
            print(f"{idx}. Product ID: {product_id} | Predicted Rating: {predicted_rating:.2f}")
        
        # Pause before displaying the next batch
        if i + batch_size < len(recommendations):
            input("\nPress Enter to see more recommendations...\n")  # Wait for user input

    # Pause before displaying recommendations for the next user
    if user_idx < num_users:
        input("\nPress Enter to see recommendations for the next user...\n")

