import streamlit as st

#If you want to run a specific .py file: 
#1. Go into the command prompt (terminal)
#2. Add the full file path to the base ---> cd to a particular folder name; ls to make sure the path is correct
#3. Use the command: streamlit run my_app.py

st.title("Hello, streamlit!")  #---> Makes header: Similar to having a single hastag in a markdown.
st.markdown("#Hello, streamlit!")


st.write("This is my first streamlit app.") #st.write puts text into paragraph size level

#Creates a button
if st.button("Click me!"):
    st.write("You clicked the button!")
else: 
    st.write("Click the button to see what happens")


###Loading our csv file
import pandas as pd 

st.subheader("Exploring Our Dataset")

#Load the CSV file 

df = pd.read_csv("data/sample_data.csv") 
#Don't copy the entire file path because the directory to this folder is already loaded.

st.write("Here's our data")
st.dataframe(df)

#Create a select box to allow people to filter for individual cities:
#st.selectbox("Directions", ["List of things to chose from"])
st.selectbox("Select a city", ["New York", "Chicago"]) 

#OR
city = st.selectbox("Select a city option two", df["City"].unique(), index = None)
#index = None -- creates the selection box so that it doesn't have a predetermined option
filtered_df = df[df["City"] == city]

st.write(f"People in {city}")
st.dataframe(filtered_df)

###Add bar chart: 
st.bar_chart(df["Salary"])

#Creating a visualization with the package seaborn: 
import seaborn as sns

box_plot1 = sns.boxplot(x = df["City"], y = df["Salary"]) #Create gragh
st.pyplot(box_plot1.get_figure()) #Add get_figure() to get the seaborn graph to work in Visual Studio


