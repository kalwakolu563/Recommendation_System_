import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# Load Dataset
df = pd.read_csv("amazon_ratings_electronics.csv")
df

print(df.columns)

# Basic info and data check
print("Data Info:")
print(df.info())

# Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())


# Basic statistics of ratings
print("\nRating Distribution:")
print(df['Rating'].describe())


# Count of unique users and products
print(f"\nNumber of Unique Users: {df['User_ID'].nunique()}")
print(f"Number of Unique Products: {df['Product_ID'].nunique()}")


# Most active users
top_users = df['User_ID'].value_counts().head(10)
print("\nTop 10 Users by Number of Ratings:")
print(top_users)


# Most rated products
top_products = df['Product_ID'].value_counts().head(10)
print("\nTop 10 Most Rated Products:")
print(top_products)

# Visualization: Rating distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Rating',hue='Rating', palette='viridis',legend=False)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# Visualization: Top 10 most active users
plt.figure(figsize=(8, 5))
top_users.plot(kind='bar', color='skyblue')
plt.title('Top 10 Most Active Users')
plt.xlabel('User ID')
plt.ylabel('Number of Ratings')
plt.xticks(rotation=45)
plt.show()


# Visualization: Top 10 most rated products
plt.figure(figsize=(8, 5))
top_products.plot(kind='bar', color='salmon')
plt.title('Top 10 Most Rated Products')
plt.xlabel('Product ID')
plt.ylabel('Number of Ratings')
plt.xticks(rotation=45)
plt.show()

# Convert IDs to categorical type
df['User_ID'] = df['User_ID'].astype('category')
df['Product_ID'] = df['Product_ID'].astype('category')

# Create a sparse matrix directly using category codes
sparse_matrix = coo_matrix((
    df['Rating'].values, 
    (df['User_ID'].cat.codes.values, df['Product_ID'].cat.codes.values)
))

# Convert to CSR format for efficient operations
sparse_matrix = csr_matrix(sparse_matrix)

user_item_matrix = pd.DataFrame.sparse.from_spmatrix(
    sparse_matrix, 
    index=df['User_ID'].cat.categories, 
    columns=df['Product_ID'].cat.categories
)

# Convert data type to float32 (optional, only if required)
user_item_matrix = user_item_matrix.astype('float32')  
print(user_item_matrix.head())

user_id = 'A001944026UMZ8T3K5QH1'  
print(user_item_matrix.loc[user_id])

product_id = '132793040'  
print(user_item_matrix[product_id])

# pivot_df = user_item_matrix.reset_index()
# print(pivot_df.head())

# Apply TruncatedSVD for Dimensionality Reduction
svd = TruncatedSVD(n_components=50, random_state=42)
latent_matrix = svd.fit_transform(user_item_matrix)

# Fit Nearest Neighbors Model
nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
nn_model.fit(latent_matrix)

# Function to Recommend Products for a User
def recommend_products(user_id, user_item_matrix, latent_matrix, nn_model, top_n=10):
    if user_id not in user_item_matrix.index:
        print("User not found.")
        return []
    
    user_idx = list(user_item_matrix.index).index(user_id)
    distances, indices = nn_model.kneighbors([latent_matrix[user_idx]])
    similar_users = [user_item_matrix.index[i] for i in indices[0][1:]]  # Get top 5 similar users
    
    recommended_products = set()
    for similar_user in similar_users:
        top_products = user_item_matrix.loc[similar_user].sort_values(ascending=False).index[:top_n]
        recommended_products.update(top_products)
    
    return list(recommended_products)[:top_n]


    # Example: Get recommendations for a specific user
example_user = df['User_ID'].iloc[0]  
recommended_products = recommend_products(example_user, user_item_matrix, latent_matrix, nn_model)

print(f"\nTop Recommendations for User {example_user}:")
print(recommended_products)

#get the first 5 user recomendations

# Get the first 5 unique users from the dataset
example_users = df['User_ID'].unique()[:5]

# Loop through each user and get recommendations
for user in example_users:
    recommended_products = recommend_products(user, user_item_matrix, latent_matrix, nn_model)
    print(f"\n🔹 Top Recommendations for User {user}:")
    print(recommended_products)
