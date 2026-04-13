#Import necessary packages

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_curve, roc_auc_score, auc, f1_score, 
                             precision_score, recall_score)
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ------------------------------------
# Page Setup
# ------------------------------------
st.set_page_config(
    page_title="Supervised Machine Learning",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Add a color scheme
st.markdown(
    """
    <style>
    /* Main background - Light Cool Grey */
    .stApp {
        background-color: #F8F9FB;
    }

    /* Sidebar background - Soft White */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E6E9EF;
    }

    /* Text color - Charcoal for high readability */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #262730 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------
# Application Information
# ------------------------------------

# Describe application
st.title("Supervised Machine Learning: Decision Tree & Logistic Regression :dart:")
st.markdown("""
### About this Application
This interactive application shows the performance of two supervised machine learning models, a decision tree classifier and logistic regression, based on user's selected parameters. 
            A user may upload a dataset or utilize one of the two demo datasets to explore how adjusting model imputs affects scoring outcomes such as accuaracy. 

**With this application, a user may:**
* Choose different model criterion to prioritize 
* Adjust model hyperparameters 
* Visualize a model's performance
* Receive computer generated recommendations on how to create the optimal decision tree model
""")

# Creates a division between the analysis and description: 
st.divider()

# ------------------------------------
# Data Preprocessing
# ------------------------------------

#Data preprocessing (logistic regression)
def preprocess_logistic(df, target_feature): 
    # Drops missing values 
    df_clean = df.dropna().reset_index(drop = True)

    # Encode Target Variable: 
    le = LabelEncoder()
    df_clean[target_feature] = le.fit_transform(df_clean[target_feature])

    # Identify columns where the datatype is 'object' or 'category'
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_feature in categorical_cols:
        categorical_cols.remove(target_feature)

    # Identify columns which have a 'numeric' datatype
    numeric_cols = df_clean.select_dtypes(include= ['number']).columns.tolist()

    # Identify which columns should be scaled
    need_to_scale = [col for col in numeric_cols if df_clean[col].nunique() > 2 and col != target_feature]
    
    # Scale necessary columns
    if need_to_scale: 
            scaler = StandardScaler()
            df_clean[need_to_scale] = scaler.fit_transform(df_clean[need_to_scale])

    # Converts categorical columns into dummy variables
    if categorical_cols:
        df_clean = pd.get_dummies(df_clean, columns = categorical_cols, drop_first=True)

    # Ensure that the target feature is an integer
    df_clean[target_feature] = df_clean[target_feature].astype(int)
    
    return df_clean


# Data preprocessing (decision tree classifier regression)
def preprocess_decision_tree(df, target_feature): 
    # Drop missing values 
    df_clean = df.dropna().reset_index(drop=True)

    # Encode Target Variable: 
    le = LabelEncoder()
    df_clean[target_feature] = le.fit_transform(df_clean[target_feature])

    # Identify columns where the datatype is 'object' or 'category'
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_feature in categorical_cols:
        categorical_cols.remove(target_feature)
    
    # Converts categorical columns into dummy variables
    df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

    # Ensure that the target feature is an integer
    df_clean[target_feature] = df_clean[target_feature].astype(int)

    return df_clean

# ------------------------------------
# Split Data 
# ------------------------------------

def split(df_clean, selected_features, target_feature, test_size=0.2, random_state=42):
    X = df_clean[selected_features]
    y = df_clean[target_feature]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ------------------------------------
# Training Models 
# ------------------------------------
# Create logistic regression model: 
def train_logistic_reg(X_train, y_train): 
    #Train the logistic regression model
    model_log = LogisticRegression()
    model_log.fit(X_train, y_train)
 
    return model_log

# Create classification model: 
def train_classification(X_train, y_train, criterion, max_depth, min_samples_split, min_samples_leaf): 
    
    #Train the classification model
    model_class = DecisionTreeClassifier(criterion= criterion, 
                                         max_depth= max_depth, 
                                         min_samples_split= min_samples_split, 
                                         min_samples_leaf= min_samples_leaf,
                                         random_state= 42)
    model_class.fit(X_train, y_train)
 
    return model_class

# ------------------------------------
# Evaluating Models 
# ------------------------------------

# Evaluate logistic regression model: 

def evaluate_logistic_reg(y_test, model_log, X_test): ###Double check selected features!!!!
   
   # Predict on test data
    y_pred = model_log.predict(X_test)

    # Display classification report
    st.write("***Classification Report:***")
    st.code(classification_report(y_test, y_pred))

    # Generate confusion matrix
    fig, ax = plt.subplots(figsize = (3,2)) 
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontsize = 5)
    ax.set_xlabel('Predicted', fontsize = 4)
    ax.set_ylabel('Actual', fontsize = 4)
    st.pyplot(fig)

# Extract the coefficients from the log model

def log_coef(model_log, selected_features):
    # Extract and print the coefficients
    coef_df = pd.DataFrame(model_log.coef_[0], 
                     index = selected_features, 
                     columns = ['Coefficient'])
    st.write("**Model Coefficients:**")
    st.write(coef_df)


# Tuning the Decison Tree model --> find the best tree

def find_best_tree(X_train, y_train, X_test, scoring_choice):
    # Define parameter grid
    param_grid = {
        "criterion" : ["gini", "entropy", "log_loss"],
        "max_depth" : range(1, 11),
        "min_samples_split" : range(2, 11, 2),
        "min_samples_leaf" : range(1, 11)
    }
    # Initialize decision tree classifier
    dtree = DecisionTreeClassifier(random_state = 42)
    # Set up grid search cv
    grid_search = GridSearchCV(estimator = dtree,
                               param_grid = param_grid,
                               cv = 5,
                               scoring = scoring_choice)
    # Fit grid search cv to the training data
    grid_search.fit(X_train, y_train)
    # Get best parameters
    best_params = grid_search.best_params_
    # Get best estimator
    best_dtree = grid_search.best_estimator_
    # Predict on the test set
    y_pred = best_dtree.predict(X_test)
    return y_pred, best_params, best_dtree

# Evaluate decision tree model: 

def evaluate_class(best_dtree, X_test, y_test):
   
   # Predict on test data
    y_pred = best_dtree.predict(X_test)

    # Display classification report
    st.write("***Tuned Classification Report:***")
    st.code(classification_report(y_test, y_pred))

    # Generate confusion matrix
    fig, ax = plt.subplots(figsize = (3, 2)) 
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontsize = 5)
    ax.set_xlabel('Predicted', fontsize = 4)
    ax.set_ylabel('Actual', fontsize = 4)
    st.pyplot(fig)


# Calculate selected scoring metric
def calculate_metrics(y_test, y_pred, scoring_choice):

    # Check if dataset is multiclass (more than 2 unique values)
    is_multiclass = len(np.unique(y_test)) > 2
    avg_method = 'weighted' if is_multiclass else 'binary'

    # Connects each metric to its calculations
    if scoring_choice == "accuracy":
        score = accuracy_score(y_test, y_pred)
    elif scoring_choice == "f1":
        score = f1_score(y_test, y_pred, average=avg_method)
    elif scoring_choice == "precision":
        score = precision_score(y_test, y_pred, average=avg_method)
    elif scoring_choice == "recall":
        score = recall_score(y_test, y_pred, average=avg_method)
    
    st.metric(label=f"{scoring_choice.capitalize()} Score", value=f"{score:.2f}")


# Plot tuned decision tree: 
def plot_tuned_tree(best_dtree, X_train, df_tree, target_feature):
    unique_classes = [str(x) for x in sorted(df_tree[target_feature].unique())] #convert to list of strings

    # Define information to go in graph
    dot_data = tree.export_graphviz(best_dtree, 
                                    feature_names = X_train.columns, 
                                    class_names = unique_classes, 
                                    filled = True)
    
    # Generate and display decision tree graph: 
    graph = graphviz.Source(dot_data)
    st.write("***Tuned Decision Tree***")
    st.graphviz_chart(dot_data)


# Make a AUC/ROC curve ----> thank you Gemini!!
def auc_roc(model_class, X_test, y_test, best_dtree):
    # Determine if the target is binary or multiclass
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)

    if n_classes == 2:
        #Binary Target
        y_probs = model_class.predict_proba(X_test)[:, 1]
        y_probs_tuned = best_dtree.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_probs)
        roc_auc_tuned = roc_auc_score(y_test, y_probs_tuned)

        st.write(f"DT ROC AUC Score: {roc_auc:.2f}")
        st.write(f"DT-Tuned ROC AUC Score: {roc_auc_tuned:.2f}")

        #Plot binary curves
        fig, ax = plt.subplots(figsize=(6, 4))
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_probs_tuned)
        
        ax.plot(fpr, tpr, label=f'User DT (AUC = {roc_auc:.2f})')
        ax.plot(fpr_tuned, tpr_tuned, label=f'Tuned DT (AUC = {roc_auc_tuned:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_title('ROC Curve (Binary)')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    else:
        # Multiclass Target
        # Get probabilities for both the user model and the tuned model
        y_probs_user = model_class.predict_proba(X_test)
        y_probs_tuned = best_dtree.predict_proba(X_test)
        
        # Plot individual lines, turning multiclass structure into a binary system
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=unique_classes)
        
        fig, ax = plt.subplots(figsize=(8, 5)) # Slightly wider to fit the legend
        
        # Loop through each class to plot comparison lines
        for i in range(n_classes):
            # User Model Lines (Dashed)
            fpr_u, tpr_u, _ = roc_curve(y_test_bin[:, i], y_probs_user[:, i])
            auc_u = auc(fpr_u, tpr_u)
            ax.plot(fpr_u, tpr_u, linestyle='--', label=f'User Class {unique_classes[i]} (AUC = {auc_u:.2f})')
            
            # Tuned Model Lines (Solid)
            fpr_t, tpr_t, _ = roc_curve(y_test_bin[:, i], y_probs_tuned[:, i])
            auc_t = auc(fpr_t, tpr_t)
            ax.plot(fpr_t, tpr_t, label=f'Tuned Class {unique_classes[i]} (AUC = {auc_t:.2f})')
        
        # Plot styling
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title('Multiclass ROC Comparison (One-vs-Rest)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        
        # Move legend outside the plot area 
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        # Explaination of multiple classes within target variable
        st.write("""
        When using datasets with multiple classes contained within the target variable, we use the **One-vs-Rest** technique for ROC curves and AUC scores. 
        The One-vs-Rest technique chooses one class and compares it to all other classes within the target. 
        It plots a ROC curve for each class and states an overall AUC score.
        """)
        
        # Overall Score Comparison
        ovr_user = roc_auc_score(y_test, y_probs_user, multi_class='ovr')
        ovr_tuned = roc_auc_score(y_test, y_probs_tuned, multi_class='ovr')
        
        st.write(f"**Overall User AUC (OvR):** {ovr_user:.2f}")
        st.write(f"**Overall DT-Tuned AUC (OvR):** {ovr_tuned:.2f}")
        
        st.pyplot(fig, bbox_inches='tight')
        plt.close(fig)


# ------------------------------------
# Give Dataset Options 
# ------------------------------------

# Dataset Upload Option
st.sidebar.markdown("## Dataset Selection")
st.sidebar.markdown("To begin: Use a provided dataset or upload a tidy csv")
st.sidebar.markdown("#### Upload a csv file")

uploaded_file = st.sidebar.file_uploader("Personal file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

# Provided dataset option 
else:
    st.sidebar.markdown("#### Use a demo dataset")
    data_demo = st.sidebar.selectbox(label = "Demo datasets", 
                                     options=["Breast Cancer", "Wine"],
                                     index = None)
    # If choose breast cancer data, convert into a dataframe
    if data_demo == "Breast Cancer":
        raw = load_breast_cancer()
        data = pd.DataFrame(raw.data, columns=raw.feature_names)
        data['is_benign_tumor'] = raw.target

   # If choose diabetes data, convert into a dataframe
    elif data_demo == "Wine":
        raw = load_wine()
        data = pd.DataFrame(raw.data, columns=raw.feature_names)
        data['wine_class'] = raw.target
    else:
        data = None # No data selected yet

# Check if data exists before running analysis
if data is not None:
    st.write("### Data Preview")
    st.dataframe(data.head())
    
    st.write(f"**Shape:** {data.shape[0]} rows, {data.shape[1]} columns")
else:
    st.info("Please upload a CSV or select a demo dataset from the sidebar to begin.")

# ------------------------------------
# Choose the classification's model parameters
# ------------------------------------

# Select hyperparameters
st.sidebar.markdown("---")
st.sidebar.markdown('## Hyperparameter Selection')

# Select scoring metric for GridSearchCV --> map scoring options onto variables
scoring_map = {
    "Accuracy": "accuracy",
    "F1 Score": "f1",
    "Precision": "precision",
    "Recall": "recall"}

# Make dropbox for options
scoring_choice = st.sidebar.selectbox(label="Scoring metric",
                                      options=list(scoring_map.keys()),
                                      index=0)
scoring_metric = scoring_map[scoring_choice]


# Select criterion --> map criterion onto labels
criterion_map = {
    "Gini Index": "gini",
    "Entropy": "entropy",
    "Log Loss": "log_loss"}

# Make dropbox for options
selected_label = st.sidebar.selectbox("Criterion", options=list(criterion_map.keys()), index=0)
criterion = criterion_map[selected_label] # creates critieron variable

# Select hyperparameter: max depth
max_depth = st.sidebar.select_slider(label="Maximum depth of tree",
                                    options=range(1, 8),
                                    value=1)

# Select hyperparameter: min samples split
min_samples_split = st.sidebar.select_slider(label="Minimum samples to split",
                                            options=range(2,8,2),
                                            value=2)

# Select hyperparemeter: min samples leaf
min_samples_leaf = st.sidebar.select_slider(label="Minimum samples for leaf",
                                            options=range(1, 8),
                                            value=1)


# ------------------------------------
# Applying Functions to Data: 
# ------------------------------------

#Complete the tasks if there
 
if data is not None: 
    st.divider()
    st.write("### Model Configuration")
    st.markdown(""" *Note: If using the demo datasets 'Breast Cancer' or 'Wine', please use the target features 
            'is_benign_tumor' and 'wine_class' respectively.*""")
    # Feature selection: 
    col1, col2 = st.columns(2)
    with col1: 
        all_columns = data.columns.tolist()
        
        # Makes a default value for the target drop down box
        if 'is_benign_tumor' in all_columns:
            default_target = 'is_benign_tumor'
        elif 'wine_class' in all_columns:
            default_target = 'wine_class'
        else:
            default_target = all_columns[0] # Fallback

        default_idex = all_columns.index(default_target)

        # Makes drop down box
        target_feature = st.selectbox(
            label="Select Target Variable (Y)", 
            options=all_columns,
            index=default_idex)
        
    with col2: 
        selected_features = st.multiselect("Select Features (X)", [c for c in all_columns if c != target_feature])
    
    # Data Preprocessing 
    if selected_features: 
        df_log = preprocess_logistic(data[selected_features + [target_feature]].copy(), target_feature)
        df_tree = preprocess_decision_tree(data[selected_features + [target_feature]].copy(), target_feature)

    # Model Training
    if st.button("Run Analysis"):
        #Splitting Data
        X_train, X_test, y_train, y_test = split(df_tree, selected_features, target_feature)
        X_train_log, X_test_log, y_train_log, y_test_log = split(df_log, selected_features, target_feature)
        
        tab1, tab2 = st.tabs(["Decision Tree", "Logistic Regression"])
        with tab1:
            st.subheader("Decision Tree Performance")
            

            # Train the decision tree based off of user's selections
            model_class = train_classification(X_train, 
                                               y_train, 
                                               criterion,
                                               max_depth, 
                                               min_samples_split, 
                                               min_samples_leaf)
            
            # Create predictions based off of user's model:
            y_pred_user = model_class.predict(X_test)

            # Find the best decision tree: 
            y_pred_best, best_params, best_dtree = find_best_tree(X_train, y_train, X_test, scoring_metric)
            
            # Plot the user's decision tree
            st.write("""
            Decision tree classifiers allow one visualize their model via decision trees. Decision tree show the splitting criteria at each node, feature names, and class distributions within nodes. 
            
            Below is the decision tree you made:""")
            
            plot_tuned_tree(model_class, X_train, df_tree, target_feature)

            # Divide Sections
            st.divider()

            # Evaluate the decision tree model
            st.markdown("### Scoring Outcome")
            st.write("""
            To evaluate the model, use the below scoring metrics:
            * Your Selected Scoring Metric
            * Classification Report: Shows metrics such precision, which minimizes false positives, recall, which minimizes false negatives, 
                     and the F1-score, which summarizes the performance of precision and recall. 
            * Confusion matrix: Displays the how well the model did a predicting individual observations, showing true positives, 
                     true negatives, false positives, and false negatives 
                     """)

            st.write("#### **Result of your selected scoring metric:**")
            calculate_metrics(y_test, y_pred_user, scoring_metric)

            # Add a block of empty space
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

            # Evaluate the decision tree model
            evaluate_class(model_class, X_test, y_test)

            # Divide Sections
            st.divider()

            # Creating the optimal model
            st.markdown("### Creating the Optimal Model")
            st.write("""
            Using Grid Search, the computer runs through different parameter options and creates a model that reflects the best estimator.  
            See below for the computer's recommended model and how it would alter your selected scoring metric:      """)
            
            
            # State the best parameters: 
            st.write("***To make the optimal model, choose these parameters:***")
            st.write(f"Best Parameters: {best_params}")

            # Add a block of empty space
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

            # Caluate Results of Selected Scoring Metric
            st.write("***Results of your selected scoring metric using the optimal parameters:***")
            calculate_metrics(y_test, y_pred_best, scoring_metric)

            # Add a block of empty space
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            
            # Divide the sections
            st.divider()

            st.write("### **Comparing Your Model and the Computer's Model Performance**")
            st.write("""
            The ROC Curve and AUC analysis helps to evaluate the two model's performance:
            * ROC Curve: Plots the true positive rate against the false positive rate
            * AUC (Area Under the Curve): States the model's ability to differeniate between classes 
                     """)
            # Make the AUC/ROC curve
            auc_roc(model_class, X_test, y_test, best_dtree)


        
        with tab2: 
            st.subheader("Logistic Regression Performance")

            # Train the model
            model_log = train_logistic_reg(X_train_log, y_train_log)

            # Create predictions based off of user's model
            y_pred_log = model_log.predict(X_test_log)

            # Evaluate the logistic model
            st.markdown("### Scoring Outcome")
            st.write("""
            To evaluate the model, use the below scoring metrics:
            * Your Selected Scoring Metric
            * Classification Report: Shows metrics such precision, which minimizes false positives, recall, which minimizes false negatives, 
                     and the F1-score, which summarizes the performance of precision and recall. 
            * Confusion matrix: Displays the how well the model did a predicting individual observations, showing true positives, 
                     true negatives, false positives, and false negatives 
                     """)

            st.write("#### **Result of your selected scoring metric:**")
            calculate_metrics(y_test_log, y_pred_log, scoring_metric)

            # Add a block of empty space
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

            evaluate_logistic_reg(y_test_log, model_log, X_test_log)

            # Divide the sections
            st.divider()

            st.markdown("### Interpreting the Model")
            st.write("Use the below coefficients to help you better interpret your model. " \
            "In logistic regression, the coefficients represent the impact of each feature on the log-odds of the target. A positive coefficient shows that your feature increases the odds of the outcome " \
            "while a negative coefficient shows the opposite effect. The larger a coefficient is, the greater influence it has on the outcome.")
            
            # Shows model's coefficients
            log_coef(model_log, selected_features)








