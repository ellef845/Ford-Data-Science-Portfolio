# import necessary packages
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, RocCurveDisplay
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.preprocessing import LabelBinarizer 
from itertools import cycle
from sklearn.model_selection import LogisticRegression


# ------------------------------------
# Page Setup
# ------------------------------------
st.set_page_config(
    page_title="Hyperparameter Tuning: Decision Tree vs. Linear Regression",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# ------------------------------------
# Application Information
# ------------------------------------
st.title("Hyperparameter Tuning: Decision Tree vs. Linear Regression")
st.markdown("""
### About this Application
This interactive application compares the performance of two supervised machine learning models: a decision tree classifier and linear regression. 
            Users can view visualize the models' preformance under different hyperparameters. 
            Within this program you can:
            !!!!!!!!!!!!!!!!!!!!!!!!!kjfadfhakdjsfhaklsjdhfkajsdhfalkjsdhfkasjdhfakjlsdhf
            dkjfhakjsdfhakljdhfaksdjhfaksjdhfaskjdhfaksdjhfakjsdhfaksjdhfakjsdhfkajsdhfhkjasdf
- Input your own dataset or choose one of the demo datasets available.
- Choose a metric to use as the criterion for splitting the data at each node.
- Tune the different hyperparameters.
- Automatically find the best tree according to a scoring parameter of your choice.
""")

#Creates a division between the analysis and description: 
st.divider()

# ------------------------------------
# Helper Functions
# ------------------------------------

#Create logistic regression model: 
def train_logistic_reg(df, selected_features, target_feature): 
    #Extract selected features and target feature
    X = df[selected_features].values.reshape(-1, 1)
    y = df[target_feature].values.reshape(-1, 1)

    #Train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        random_state = 42)
    
    #Train the logistic regression model
    model_log = LogisticRegression()
    model_log.fit(X_train, y_train)

    # Return the trained model and test data for evaluation
    return model_log, X_test, y_test

#Create classification model: 
def train_classification(df, selected_features, target_feature): 
    




# turn sklearn toy datasets into pandas dataframes

def toy_to_df(load_function):
    bunch = load_function()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return df

# data preprocessing
def preprocess(df):
    # encode categorical variables
    categorical_cols = [col for col in df.columns
                        if (pd.api.types.is_categorical_dtype(df[col].dtype)
                        or pd.api.types.is_object_dtype(df[col].dtype))
                        and col != target]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # define features and target
    X = df.drop(target, axis=1)
    y = df[target]
    return df, X, y

# ------------------------------------
# Give Dataset Options 
# ------------------------------------

# dataset Upload Option
st.sidebar.markdown("## Dataset Selection")
uploaded_file = st.sidebar.file_uploader("#### Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

#provided dataset option 
if uploaded_file is None: 
    st.sidebar.markdown("#### Use a demo dataset")
    data_demo = st.sidebar.selectbox(label = "Demo datasets", 
                                     options=["Breast Cancer", "Diabetes"],
                                     index = None)


breast_cancer = load_breast_cancer()
diabetes = load_diabetes()
