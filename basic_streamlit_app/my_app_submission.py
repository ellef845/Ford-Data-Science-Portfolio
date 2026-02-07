#If you want to run a specific .py file: 
#1. Go into the command prompt (terminal)
#2. Add the full file path to the base ---> cd to a particular folder name; ls to make sure the path is correct
#3. Use the command: streamlit run my_app.py

import streamlit as st
import pandas as pd 

import streamlit as st

#Create a background color:


#This is a header: 
st.set_page_config(page_title='Making a Hit',  layout='wide', page_icon=':musical_note:') 


t1, t2 = st.columns((1, 0.1), gap = "small") 

t1.title("What makes a Spotify song a hit? :studio_microphone: :fire:")
t1.markdown("""
This Exploratory Data Analysis (EDA) examines how danceability and energy affect music 
popularity and help make "a hit". Utilizing tabs, users may filter the dataset by 
specific genres, albums, and artist/artist groups to visualize how danceability and energy affect musical success. 
""")


#Creates a division between the analysis and description: 
st.divider()


#Load in Data
df = pd.read_csv("data/spotify_data.csv") 

##################################################
#Add in tabs: 
tab1, tab2, tab3 = st.tabs(["Genre", "Album", "Artist"])

with tab1:
    #Description of Section
    st.markdown("### **What determines popularity within a specific music genre?**")

    #Create drop down music selection option:
    genre_select = st.selectbox('Choose a music genre', 
                                df["track_genre"].unique(), 
                                help = 'Filter report to show only one genre')

    #Filter the dataframe based on genre select: 
    genre_filtered_df = df[df['track_genre'] == genre_select]

    #Calculate unique values from filtered data: 
    genre_total_songs = genre_filtered_df['track_name'].nunique()
    genre_total_artists = genre_filtered_df['artists'].nunique()
    genre_total_albums = genre_filtered_df['album_name'].nunique()

    #Display the information in columns: 
    m1, m2, m3, m4, m5 = st.columns((1,1,1,1,1))
        
    m1.write('')
    m2.metric(label ='Total Songs in Category', value = genre_total_songs)
    m3.metric(label = 'Total Number of Albums', value = genre_total_albums)
    m4.metric(label ='Total Artists in Category', value = genre_total_artists)
    m1.write('')

    ##########
    #Make Graphs:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Danceability vs. Popularity")
        st.scatter_chart(
            genre_filtered_df,
            x="danceability",
            y="popularity",
            color= "#FF8C42", # Dots will be colored differently if the song is explicit 
            x_label="Danceability Score",
            y_label="Popularity Score"
        )
    with col2:
        st.markdown("##### Energy vs. Popularity")
        st.scatter_chart(
            genre_filtered_df,
            x="energy",
            y="popularity",
            color= "#FF8C42", 
            x_label="Energy Score",
            y_label="Popularity Score"
        )

    ##########
    #Make dataframe with top 10 most popular songs within the selected genre category. 
    st.markdown("##### Overview of Top 10 Most Popular Songs within Selected Genre")

    # Convert the boolean column "explicit" to text
    genre_display_df = genre_filtered_df.copy()
    genre_display_df['explicit'] = genre_display_df['explicit'].astype(str)

    #Create top 10 dataframe 
    top_ten_genre = (
        genre_display_df
        .sort_values("popularity", ascending=False)
        .head(10)
    )

    st.dataframe(top_ten_genre, 
                hide_index = True, 
                column_config={
                "track_id": None,     #Hide the track_id column
                "Unnamed: 0": None}   #Hide the Unnamed:0 column which functions as an index
                )



#Creates a division between the analysis and description: 


##################################################
##################################################
#Description of Section
with tab2:

    st.markdown("### **What determines popularity within a specific album?**")

    #Create drop down album selection option:
    album_select = st.selectbox('Choose an album', 
                                df["album_name"].unique(), 
                                help = 'Filter report to show only one album')

    #Filter the dataframe based on genre select: 
    album_filtered_df = df[df['album_name'] == album_select]

    #Calculate unique values from filtered data: 
    album_total_songs = album_filtered_df['track_name'].nunique()
    album_total_artists = album_filtered_df['artists'].nunique()
    album_total_albums = album_filtered_df['album_name'].nunique()

    #Display the information in columns: 
    m1, m2, m3, m4, m5 = st.columns((1,1,1,1,1))
        
    m1.write('')
    m2.metric(label ='Total Songs in Category', value = album_total_songs)
    m3.metric(label = 'Total Number of Albums', value = album_total_albums)
    m4.metric(label ='Total Artists in Category', value = album_total_artists)
    m1.write('')


    ##########
    #Make Graphs:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Danceability vs. Popularity")
        st.scatter_chart(
            album_filtered_df,
            x="danceability",
            y="popularity",
            color= "#FF8C42",
            x_label="Danceability Score",
            y_label="Popularity Score"
        )
    with col2:
        st.markdown("##### Energy vs. Popularity")
        st.scatter_chart(
            album_filtered_df,
            x="energy",
            y="popularity",
            color= "#FF8C42", 
            x_label="Energy Score",
            y_label="Popularity Score"
        )

    ##########
    #Make dataframe with top 10 most popular songs within the selected album category. 
    st.markdown("##### Overview of Top 10 Most Popular Songs within Selected Album")

    # Convert the boolean column "explicit" to text
    album_display_df = album_filtered_df.copy()
    album_display_df['explicit'] = album_display_df['explicit'].astype(str)

    #Create top 10 dataframe 
    top_ten_album = (
        album_display_df
        .sort_values("popularity", ascending=False)
        .head(10)
    )

    st.dataframe(top_ten_album, 
                hide_index = True, 
                column_config={
                "track_id": None,     #Hide the track_id column
                "Unnamed: 0": None}   #Hide the Unnamed:0 column which functions as an index
                )


#Creates a division between the analysis and description: 

##################################################
##################################################
with tab3:
    #Description of Section
    st.markdown("### **What determines popularity for specific music artist/artists?**")

    #Create drop down artist selection option:
    artist_select = st.selectbox('Choose an artist/artist duo', 
                                df["artists"].unique(), 
                                help = 'Filter report to show only one artist')

    #Filter the dataframe based on genre select: 
    artist_filtered_df = df[df['artists'] == artist_select]

    #Calculate unique values from filtered data: 
    artist_total_songs = artist_filtered_df['track_name'].nunique()
    artist_total_artists = artist_filtered_df['artists'].nunique()
    artist_total_albums = artist_filtered_df['album_name'].nunique()

    #Display the information in columns: 
    m1, m2, m3, m4, m5 = st.columns((1,1,1,1,1))
        
    m1.write('')
    m2.metric(label ='Total Songs in Category', value = artist_total_songs)
    m3.metric(label = 'Total Number of Albums', value = artist_total_albums)
    m4.metric(label ='Total Artists in Category', value = artist_total_artists)
    m1.write('')

    ##########
    #Make Graphs:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Danceability vs. Popularity")
        st.scatter_chart(
            artist_filtered_df,
            x="danceability",
            y="popularity",
            color= "#FF8C42",
            x_label="Danceability Score",
            y_label="Popularity Score"
        )
    with col2:
        st.markdown("##### Energy vs. Popularity")
        st.scatter_chart(
            artist_filtered_df,
            x="energy",
            y="popularity",
           color= "#FF8C42",   
            x_label="Energy Score",
            y_label="Popularity Score"
        )

    ##########
    #Make dataframe with top 10 most popular songs within the selected artist/artist group category. 
    st.markdown("##### Overview of Top 10 Most Popular Songs by Selected Artist/Artist Group")

    # Convert the boolean column "explicit" to text
    artist_display_df = artist_filtered_df.copy()
    artist_display_df['explicit'] = artist_display_df['explicit'].astype(str)

    #Create top 15 dataframe 
    top_ten_artist = (
        artist_display_df
        .sort_values("popularity", ascending=False)
        .head(10)
    )

    st.dataframe(top_ten_artist, 
                hide_index = True, 
                column_config={
                "track_id": None,     #Hide the track_id column
                "Unnamed: 0": None}   #Hide the Unnamed:0 column which functions as an index
                )





