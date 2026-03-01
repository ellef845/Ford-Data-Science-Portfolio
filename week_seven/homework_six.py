# Import neccesary libraries
import pandas as pd
import numpy as np
import streamlit as st         # Framework for building interactive web apps

########Part One:
###1. Load the California Housing dataset from sklearn.datasets.

from sklearn.datasets import fetch_california_housing

# Load the housing dataset
housing = fetch_california_housing()

###2. Create a Pandas DataFrame for the features and a Series for the target variable (med_house_value). 
X = pd.DataFrame(housing.data, columns = housing.feature_names)
y = pd.Series(housing.target, name = 'med_house_value')

# ------------------------------------------------------------------------------
# Display Summary Statistics
# ------------------------------------------------------------------------------
#Show first five rows in df
st.write("**First Five Data Entries**")
st.dataframe(X.head(5))

# Show key statistical measures like mean, standard deviation, etc.
st.write("**Summary Statistics**")
st.dataframe(X.describe())

# ------------------------------------------------------------------------------
# Check for Missing Values
# ------------------------------------------------------------------------------
# Display the count of missing values for each column.
st.write("**Number of Missing Values by Column**")
st.dataframe(X.isnull().sum()) #Blank boxes mean the entry is FALSE (i.e. there is data)
                              #Sum tells us the number of null values per column 

print(X.columns)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Part 2: Linear Regression on Unscaled Data (30 points)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Import linear regression class
from sklearn.linear_model import LinearRegression #If you have questions, visit the API documentation for https://scikit-learn.org/stable/api/sklearn.linear_model.html

# Import model splitting class
from sklearn.model_selection import train_test_split #train_test_split makes a random split 

# Split and unpack the raw data (80% training, 20% testing)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) #use random_state to make results replicable for other users
