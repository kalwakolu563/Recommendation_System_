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

# Convert IDs to categorical type
df['User_ID'] = df['User_ID'].astype('category')
df['Product_ID'] = df['Product_ID'].astype('category')

# Convert Ratings to float32 for memory efficiency
df['Rating'] = df['Rating'].astype('float32')

# Create a sparse matrix directly using category codes
sparse_matrix = coo_matrix((
    df['Rating'].values, 
    (df['User_ID'].cat.codes.values, df['Product_ID'].cat.codes.values)
))

# Convert to CSR format for efficient operations
sparse_matrix = csr_matrix(sparse_matrix)

# Convert sparse matrix to Pandas DataFrame
user_item_matrix = pd.DataFrame.sparse.from_spmatrix(
    sparse_matrix, 
    index=df['User_ID'].cat.categories, 
    columns=df['Product_ID'].cat.categories
)

# Convert data type to float32 (optional, only if required)
user_item_matrix = user_item_matrix.astype('float32')  
print(user_item_matrix.head())

# Check if the User exists before lookup
user_id = 'A001944026UMZ8T3K5QH1'
if user_id in user_item_matrix.index:
    print(user_item_matrix.loc[user_id])
else:
    print(f"User {user_id} not found in dataset.")

# Check if the Product exists before lookup
product_id = '132793040'
if product_id in user_item_matrix.columns:
    print(user_item_matrix[product_id])
else:
    print(f"Product {product_id} not found in dataset.")

# Convert user-item matrix to a sparse matrix before SVD
sparse_user_item_matrix = csr_matrix(user_item_matrix)

# Apply TruncatedSVD for Dimensionality Reduction
svd = TruncatedSVD(n_components=50, random_state=42)
latent_matrix = svd.fit_transform(sparse_user_item_matrix)

# Fit Nearest Neighbors Model
nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
nn_model.fit(latent_matrix)

# Function to Recommend Products for a User
def recommend_products(user_id, user_item_matrix, latent_matrix, nn_model, top_n=10):
    if user_id not in user_item_matrix.index:
        print("User not found.")
        return []
    
    # Efficiently get user index
    user_idx = user_item_matrix.index.get_loc(user_id)

    # Find similar users
    distances, indices = nn_model.kneighbors([latent_matrix[user_idx]])
    similar_users = [user_item_matrix.index[i] for i in indices[0][1:]]  # Get top 5 similar users
    
    recommended_products = set()
    for similar_user in similar_users:
        top_products = user_item_matrix.loc[similar_user].sort_values(ascending=False).index[:top_n]
        recommended_products.update(top_products)
    
    return list(recommended_products)[:top_n]


# Example: Get recommendations for the first 5 users
example_users = df['User_ID'].unique()[:5]  # Get the first 5 user IDs
for example_user in example_users:
    recommended_products = recommend_products(example_user, user_item_matrix, latent_matrix, nn_model)
    print(f"\nTop Recommendations for User {example_user}:")
    print(recommended_products)
