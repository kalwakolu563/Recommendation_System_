
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("amazon_ratings_electronics.csv")
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
sns.countplot(data=df, x='Rating', hue='Rating', palette='viridis', legend=False)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

time.sleep(2)


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