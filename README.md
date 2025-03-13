# Recommendation_System

Product Recommendation System Using Machine Learning
Overview
This project implements a Product Recommendation System using collaborative filtering techniques and machine learning models. 
It leverages Singular Value Decomposition (SVD) and K-Nearest Neighbors (KNN) from the Surprise and scikit-learn libraries to recommend products based on user preferences.

The system is designed to analyze user-product interactions and suggest relevant items for users based on their past preferences or similar user behaviors. This is commonly used in e-commerce platforms to enhance customer experience by recommending products they are likely to purchase.

The project includes:
✅ User-based and item-based collaborative filtering
✅ Matrix factorization using SVD
✅ Nearest Neighbors (KNN) for similarity-based recommendations
✅ Exploratory Data Analysis (EDA) for understanding data distributions

Project Structure
📂 rc2.py – Implements TruncatedSVD and Nearest Neighbors for recommendation modeling.
📂 rc_system.py – Contains a function to get top recommendations for a single user.
📂 rc_version2.py – Allows recommending products for a sample user input.
📂 eda.py – Performs exploratory data analysis (EDA) using Matplotlib and Seaborn.
📂 requirements.txt – Lists all the necessary libraries for the project.
📂 read_me.txt – Provides steps to set up and activate the environment and install dependencies.

Installation and Setup
1️⃣ Create a Virtual Environment (Recommended)
python -m venv env

2️⃣ Activate the Virtual Environment
On Windows:
.\env\Scripts\activate
On macOS/Linux:
source env/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt


Usage
1️⃣ Running Exploratory Data Analysis (EDA)
To visualize the dataset and understand user-product interactions, run:
python eda.py

2️⃣ Generating Recommendations
Get recommendations for a single user:
python rc_system.py
This will return the top recommended products for a given user based on past purchases and interactions.

Test recommendations for a sample user:
python rc_version2.py
This script will take a sample user ID and suggest relevant products.

Using Nearest Neighbors and SVD for recommendations:
python rc2.py
This script applies dimensionality reduction techniques to generate recommendations.

Example Usage
🔹 Example User Input:
user_id = 12345
🔹 Recommended Products Output:
["Product A", "Product B", "Product C", "Product D"]
