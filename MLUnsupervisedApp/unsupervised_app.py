# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.datasets import load_iris

# ------------------------------------
# Page Setup
# ------------------------------------
st.set_page_config(
    page_title="Unsupervised Machine Learning",
    page_icon=":mag:",
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
st.title("Unsupervised Machine Learning: Principal Component Analysis and K-Means 💻")
st.markdown("""
### About this Application
This interactive application allows users to experiment with two unsupervised machine learning models: Principal Component 
            Analysis (PCA) and K-Means. PCA focuses on dimensionality reduction, reducing datasets with
            a large number of features into summarized components (groupings of variables) while K-Means breaks down datasets
            into k number of clusters, based on common characteristics shared by observations.
            Using this application, a user may upload a dataset or utilize one of the two demo datasets to explore how adjusting different 
            model imputs and hyperparameters affects model training and performance. 

**With this application, a user may:**

* Experiment with unsupervised machine learning models, PCA and K-Means, using their own dataset 
* Adjust K-Means model hyperparameters: the number of desired clusters and maximum model iterations
* Visualize a model's performance feedback via graphs and other metrics
 
            
*****Note: For both Principal Component Analysis (PCA) and K-Means, a target feature is unnecessary to calculate model results.
    However, this app requires the insertion of a target feature to assist with data visualization. Using a dataset
    with a target feature, allows you to see whether or not PCA and K-means worked through the creation of 2D plots! 
    We mark and exclude the target feature from the overall analysis to prevent data leakage.***
""")

# Creates a division between the analysis and description: 
st.divider()

# ------------------------------------
# Data Preprocessing
# ------------------------------------

#Data preprocessing (PCA)

def preprocess_PCA(df, target_feature):
    # Drop NA values
    df_clean = df.dropna().reset_index(drop=True)

    # Encode Target variable
    le = LabelEncoder()
    df_clean[target_feature] = le.fit_transform(df_clean[target_feature])

    # Get dummies
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_feature in categorical_cols:
        categorical_cols.remove(target_feature)
    
    df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

    # Identify numeric features (excluding target)
    numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    features_to_scale = [col for col in numeric_cols if col != target_feature]
    
    # Scale features
    if features_to_scale: 
        scaler = StandardScaler()
        df_clean[features_to_scale] = scaler.fit_transform(df_clean[features_to_scale])

    df_clean[target_feature] = df_clean[target_feature].astype(int)

    return df_clean

def naming_columns(df_clean, target_feature): 
    # Make a variable for feature names: 
    feature_names = df_clean.drop(columns = [target_feature]).columns.tolist()

    #Make a variable for target names: 
    target_names = df_clean[target_feature].unique().tolist()

    return (feature_names, target_names)


# ------------------------------------
# Compute PCA
# ------------------------------------

#Reduce the data to two components for visualization and further analysis
def reduce_data(X_std): 
    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X_std)
   
    return (X_pca, pca)

#State the explained and cumulative_variance
def pca_var(pca): 
    # Calculate variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    #Print values for explained and cumulative variance:
    st.write(f"**Explained Variance Ratio:** {explained_variance.round(3)}")
    st.write(f"**Cumulative Explained Variance:** {cumulative_variance[-1]:.2%}")

    


# ------------------------------------
# Visualizing PCA Results
# ------------------------------------

# Scatter Plot of PCA Scores: 

def PCA_scatter(X_pca, target_names, y):
    fig, ax = plt.subplots(figsize=(7, 5), dpi = 80)
    # Generate a color palette based on how many targets exist
    colors = sns.color_palette("viridis", len(target_names))
    
    for i, (target_name, color) in enumerate(zip(target_names, colors)):
        ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                    color=color, alpha=0.7, label=target_name, 
                    edgecolor='none', s=5) 
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: 2D Projection of Selected Data')
    plt.legend(loc = 'best')
    plt.grid(True)
    st.pyplot(fig)

### PCA Loadings: Horizaontal Grouped Bar Chart 

def PCA_load_plot(pca, feature_names):

    #Build a dataframe from pca.components_ 
    #Let each row be a principal component and each column be a feature's loading weight. 
    loadings_df = pd.DataFrame(
        pca.components_, 
        columns = feature_names, 
        index = [f'PC{i+1}' for i in range(pca.n_components)]
    )
    #Create positions for horizontal bars, ensuring that each feature gets it own row. 

    features = loadings_df.columns.to_list()
    y_pos = np.arange(len(features))
    bar_height = 0.3

    fig, ax = plt.subplots(figsize = (3,4)) 

    # Plot PC1 and PC2 loadings side by side for each feature.
    # bar_height/2 puts blank, vertical space between every listing, preventing overlap
    ax.barh(y_pos + bar_height/2, loadings_df.loc['PC1'], bar_height,
            label='PC1', color='#1b2a4a', edgecolor='none')
    ax.barh(y_pos - bar_height/2, loadings_df.loc['PC2'], bar_height,
            label='PC2', color='#c5a829', edgecolor='none')

    # Label the y-axis with feature names and the x-axis with loading weight
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Loading Weight')
    ax.set_title('PCA Loadings (how each stat contributes)', fontweight='bold', loc='left')

    # Add a vertical reference line at 0 so we can see positive vs. negative loadings
    ax.axvline(0, color='grey', linewidth=0.8)

    ax.legend(loc='upper right', frameon=True)
    ax.invert_yaxis()          # Put the first feature at the top of the chart
    ax.grid(axis='x', alpha=0.3)
    ax.set_frame_on(False)     # Remove the border box for a cleaner look
    st.pyplot(fig)

#Create a combined plot of cumlative variance explained and the variance for each individual component
#----> helps to determine how many components to use 

def PCA_var_plot(X_std): 
    
    # Calculates the PCA for different components
    max_components = min(X_std.shape[1], 15)
    pca_full = PCA(n_components = max_components).fit(X_std)
   
   # Calculates the explained variance for component options
    explained = pca_full.explained_variance_ratio_ * 100
    components = np.arange(1, len(explained) + 1)
    cumulative = np.cumsum(explained)
    
    # Creates the combined plot
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Bar plot for individual variance explained
    bar_color = 'steelblue'
    ax1.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Individual Variance Explained (%)', color=bar_color)
    ax1.tick_params(axis='y', labelcolor=bar_color)
    ax1.set_xticks(components)
    ax1.set_xticklabels([f"PC{i}" for i in components])

    # Add percentage labels on each bar
    for i, v in enumerate(explained):
        ax1.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

    # Create a second y-axis for cumulative variance explained
    ax2 = ax1.twinx()
    line_color = 'crimson'
    ax2.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color)
    ax2.set_ylim(0, 100)

    # Remove grid lines
    ax1.grid(False)
    ax2.grid(False)

    # Combine legends from both axes and position the legend inside the plot (middle right)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

    plt.title('PCA: Variance Explained', pad=20)
    st.pyplot(fig)

# ------------------------------------
# Compute K-Means
# ------------------------------------

#Define a function which computes K-Means

def compute_kmeans(k_clusters, max_runs, X_std):
    kmeans = KMeans(n_clusters = k_clusters, max_iter = max_runs, random_state = 42)
    clusters = kmeans.fit_predict(X_std)

    return(clusters)

# ------------------------------------
# Evaluate K-Means Model
# ------------------------------------

# Define a function which calculates model accuracy
def kmeans_accuracy(y, clusters): 
    kscore = accuracy_score(y, clusters)

    st.write("Accuracy Score: {:.2f}%".format(kscore * 100))

# Define a function to calculate silhouette scores

def calculate_silhouette_score(X_std):
    wcss = []               # Within-Cluster Sum of Squares for each k
    silhouette_scores = []  # Silhouette scores for each k
    ks = range(2, 11) #states range of cluster values

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_std)
        wcss.append(km.inertia_)  # inertia: sum of squared distances within clusters
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X_std, labels))
    return(ks, wcss, silhouette_scores)

# ------------------------------------
# Visualizing KMeans Results
# ------------------------------------

def kmeans_elbow_sil_plot(ks, wcss, silhouette_scores):

    # Create figure object
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the Elbow Method result on the first axis (ax1)
    ax1.plot(ks, wcss, marker='o')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)

    # Plot the Silhouette Score result on the second axis (ax2)
    ax2.plot(ks, silhouette_scores, marker='o', color='green')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score for Optimal k')
    ax2.grid(True)

    plt.tight_layout()

    # Dsplay the figure
    st.pyplot(fig)

def kmeans_cluster(X_std, clusters, k_clusters):
    #Cluster Plot
    X_pca_viz, _ = reduce_data(X_std)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique cluster IDs (e.g., [0, 1, 2])
    unique_clusters = np.unique(clusters)

    # Makes distinct colors for each clusters
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

    # 4. Plot each cluster separately to apply a label
    for i, cluster_id in enumerate(unique_clusters):
        ax.scatter(
        X_pca_viz[clusters == cluster_id, 0], 
        X_pca_viz[clusters == cluster_id, 1], 
        color=colors[i],
        label=f'Cluster {cluster_id}',  # This creates the label for the legend
        alpha=0.7,
        edgecolors='w',
        s=40
                    )

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'2D Scatter Plot of K-Means Clustering (k={k_clusters}) Using PCA')
    plt.legend(title="Assigned Clusters", loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# ------------------------------------
# Sidebar Navigation  
# ------------------------------------
st.sidebar.title("Navigation")
st.sidebar.divider()

st.sidebar.markdown("### Model Selection")
#Allow users to choose a model
analysis_mode = st.sidebar.radio("Choose Analysis Type", ["PCA (Dimensionality Reduction)", "K-Means (Clustering)"])

st.sidebar.divider()
with st.sidebar.expander("K-Means Settings 🛠️", expanded=True):
    k_clusters = st.select_slider(
        label="Desired Clusters (k)",
        options=range(2, 8),
        value=2
    )
    max_runs = st.select_slider(
        label="Maximum Iterations",
        options=range(200, 501, 50),
        value=200
    )

st.sidebar.divider()

# Dataset Upload Option
st.sidebar.markdown("### Dataset Selection")
st.sidebar.markdown("To begin, use a provided dataset or upload a tidy, tabular csv.")
st.sidebar.markdown("#### Upload a csv file")

uploaded_file = st.sidebar.file_uploader("Personal file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

# Provided dataset option 
else:
    st.sidebar.markdown("#### Use a demo dataset")
    data_demo = st.sidebar.selectbox(label = "Demo datasets", 
                                     options=["Titanic", "Iris"],
                                     index = None)
    # If choose breast cancer data, convert into a dataframe
    if data_demo == "Titanic":
        raw = sns.load_dataset('titanic')
        data = raw.copy()

   # If choose diabetes data, convert into a dataframe
    elif data_demo == "Iris":
        raw = load_iris()
        data = pd.DataFrame(raw.data, columns=raw.feature_names)
        data['species'] = raw.target
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
# Choose the PCA's model parameters
# ------------------------------------

# Select hyperparameters
st.sidebar.markdown("---")
st.sidebar.markdown('')
st.sidebar.markdown('')

# ------------------------------------
# Set up Model Selection: 
# ------------------------------------

#Complete the tasks if there
 
if data is not None: 
    st.divider()
    st.write("### Model Configuration")
    st.markdown("""
    ****Note: If using the demo datasets 'Titanic' or 'Iris', please use the target features 'survived' and 'species' respectively.***
                """)

# Target selection: 
    col1, col2 = st.columns(2)
    with col1: 
        all_columns = data.columns.tolist()

        # Set a default value for the target drop-down box
        if 'species' in all_columns:
            default_target = 'species'
        elif 'survived' in all_columns:
            default_target = 'survived'
        else:
            default_target = all_columns[0]  # Fallback



        # Get the index of the default target
        default_index = all_columns.index(default_target)

        # Create the drop-down box
        target_feature = st.selectbox(
            label="Select Target Variable (Y)", 
            options=all_columns,
            index=default_index
        )
                
        with col2: 
            #Create a list of all features options 
            feature_names = st.multiselect("Select Features (X)", [c for c in all_columns if c != target_feature])
else: 
    st.info("Please upload a tabular, tidy CSV or select a demo dataset from the sidebar to begin.")


# ------------------------------------
# Preprocess Data and Create Button to Run Model: 
# ------------------------------------
 
#Make a button to run the analysis: 

if st.button(f"Run Analysis {analysis_mode}"):
        st.divider()
        ######### Data Preprocessing 
    # Make sure user has selected features
        if not feature_names:
            st.error("Please select at least one feature (X) to run the analysis.")
        else:
            # Filter the data to ONLY include selected features and target variable
            cols_to_use = feature_names + [target_feature]
            data_subset = data[cols_to_use].copy()

            #Process that subset
            df_processed = preprocess_PCA(data_subset, target_feature)

            # Split the data
            X_std = df_processed.drop(columns=[target_feature])
            y = df_processed[target_feature]

    # ------------------------------------
    # Run PCA Model: 
    # ------------------------------------
        if analysis_mode == "PCA (Dimensionality Reduction)":
            st.header("Principal Component Analysis :chart_with_upwards_trend:")
            
            #Add note stating that this app makes PCA only use two components 
            st.write("****Note: This app limits PCA to two components for graphing purposes. Using two components allows for " \
            "the creation of 2D figures.***")

            #Reduce the dataset to user's selected number of components
            X_pca, pca_model = reduce_data(X_std)

            #Create tabs: 
            tab1, tab2, tab3, tab4 = st.tabs(["Variance Analyis", "2D Projection", "Feature Loadings", "Scree Plot"])

            with tab1:
                #Calucate the explained and cumulative variance
                st.write("#### Variance Summary")
                pca_var(pca_model)
                
                #Interpretation: 
                st.write("***Interpretation:*** The explained variance ratio shows how much of the total " \
                "dataset's variance is accounted for by each principal component. The cumulative explained " \
                "variance states how much variance you gain as you add additional components. " \
                "By multiplying both the explained variance ratio and the cumulative ratio by 100, " \
                "we can see what percent of each type of variance is explained by our model. ")

            with tab2: 
                st.subheader("2D Projection")
                # Make the scatter plot
                PCA_scatter(X_pca, naming_columns(df_processed, target_feature)[1], y)
                #Interpretation: 
                st.write("#### Interpretation")
                st.write("A scatter plot of PCA scores shows where observations lie on the new coordinate system created by our two components (or groups). By using datasets with a target variable, " \
                "we can see whether or not PCA correctly stratified the observations into distinct groups.")
            
            with tab3:
                # Make load plot
                st.subheader("Feature Loadings")
                PCA_load_plot(pca_model, X_std.columns.tolist())
                #Interpretation: 
                st.write("#### Interpretation")
                st.write("A PCA loadings chart depicts the importance that each feature has to an individual component. " \
                "Loadings can be either positive or negative. A variable with a positive loading is positively correlated with a component " \
                "while a variable with a negative loading indicates that it is inversely correlated with a component.")
                
            with tab4:
                #Make an explained variance plot
                st.subheader("Variance Explained (Scree Plot)")
                PCA_var_plot(X_std)

                #Interpretation: 
                st.write("#### Interpretation")
                st.write("A scree plot shows how many components are needed for the ideal PCA analysis on your dataset. " \
                "The model's cumulative explained variance is shown on the y-axis while the number of component options is shown on the x-axis. " \
                "An elbow in the cumulative variance curve indicates that an additional component is no longer contributing to the overall explained variance of the model. " \
                "Choosing the component at the elbow maximizes the model's overall effectiveness.")
            
# ------------------------------------
# Run K-Means Model: 
# ------------------------------------

        elif analysis_mode == "K-Means (Clustering)":
            st.header("K-Means Clustering 🎯")
            
            # Compute K-Means and silhouette score
            clusters = compute_kmeans(k_clusters, max_runs, X_std)
            ks, wcss, silhouette_scores = calculate_silhouette_score(X_std)

            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Model Accuarcy", "2D Cluster Visulization", "Optimal K Analysis Plots"])

            with tab1:
                st.write("#### Accuracy Summary")
                kmeans_accuracy(y, clusters)
                st.write("""***Interpretation:*** Measuring the K-means model's accuracy, we can see how well the model did at '
                'predicting our target variable. A higher accuracy score means that the model's generated clusters closely align with 
                the true data labels, while a lower accuracy score, indicates the opposite. """)

        
            with tab2:
                st.subheader("Cluster Visualization")
                kmeans_cluster(X_std, clusters, k_clusters)
                st.write("#### Interpretation")
                st.write("""A 2D scatter plot of clustering results using PCA helps to show how K-means divides up a dataset into clusters.
                         If dots are distinctly separated into groups of colors, the model has a strong cluster quality. If
                         many different color observations overlap or mix, the model has poor cluster quality, 
                         indicating the incorrect quantity of clusters or suboptimal feature selection. 
                         The plot also assists with identifying dataset outliers. To create the plot, we reduce the data to two components then plot PCA scores. 
                         The color of clusters pertain to observations' different cluster assignments.""")

            with tab3:
                st.subheader("Elbow Method and Silhouette Score Plots")
                kmeans_elbow_sil_plot(ks, wcss, silhouette_scores)
                st.write("#### Interpretation")
                st.markdown("""
                * **Elbow Method:** The elbow method helps to recommend the optimal number of clusters. \
                            It plots the Within-Cluster Sum of Squares (WCSS) on the y-axis and different \
                            values of k, number of clusters, on the x-axis. By searching for the "elbow" in the graph, where there is a bend 
                            or change in the rate of decrease,
                            we can find an optimal value of k.  
                * **Silhouette Score:** The silhouette score indicates how similar an observation in its cluster compared to other clusters.
                """)