### Description:
This folder encompasses all files for my Unsupervised Machine Learning Application Project. 
My app allows users to interact and experiment with two unsupervised machine learning models, Principal Component Analysis (PCA) and K-Means. Users may upload their own dataset or use provided demo datasets to select features to use in their model. They may also observe how altering hyperparameters impacts model training and performance.

### App Features: 
PCA and K-Means models are utilized to learn more about unlabeled datasets. PCA focuses on dimensionality reduction, reducing datasets witha large number of features into summarized components (groupings of variables) while K-Means breaks down datasets into k number of clusters, based on 
common characteristics shared by observations.

Within the application, users may select different model features to use in calculations as well as adjust hyperparameters of the K-Means model (maximum iterations and number of desired clusters (k)).
Selection of model feaures occurs under "Model Configuration" through dropdown boxes while hyperparamters are adjusted using sidebar sliders. 

After a user personalizes and runs their model, they may receive scoring metrics and model feedback via visualizations. With the PCA model, users can observe their model's explained and cummulative variance, a 2D projection of their results, feature loadings, and a scree plot. Using the K-Means model, users can see model accuracy, 2D Cluster Visualization, and plots which suggest the optimal value of clusters (k). 

### Data: 
The provided demo datasets, ["Iris"](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) is sourced from scikit-learn while ["Titanic"]([https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv)), is from the Seaborn library. The Iris dataset contains information regarding species of irises, possessing features such as sepal width and length and petal width and length. The Titanic dataset contains information regarding titanic passengers. Some of its features include passager sex, paid fare, ship class, and age.

### Directions on how to run the streamlit app: 
To run the app locally:
1. Download the MLUnsupervisedApp folder.
2. Make sure you have the packages numpy, pandas, matplotlib, seaborn, scikit-learn, and streamlit installed within your python environment.
3. Enter "streamlit run unsupervised_app.py" into your terminal. 

You may also run the app via this ["link"](). 

### References: 
For project references, I used ["Data Wrangling with pandas Cheat Sheet"](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) made by http://pandas.pydata.org as well as course notes on ["PCA"](https://github.com/ellef845/Ford-Data-Science-Portfolio/blob/main/week_twelve/IDS_12_2_FINAL.ipynb) and ["K-Means"](https://github.com/ellef845/Ford-Data-Science-Portfolio/blob/main/week_thirteen/IDS%2013_1_FINAL.ipynb) created by David Smiley.




