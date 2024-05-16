import streamlit as st
import pandas as pd
from pytube import YouTube
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("movie_metadata.csv")

# Data preprocessing
df['gross'].fillna(df['gross'].mean(), inplace=True)
df['budget'].fillna(df['budget'].mean(), inplace=True)
df.dropna(inplace=True)
df['main_genre'] = df['genres'].apply(lambda x: x.split('|')[0] if '|' in x else x)
df['success_label'] = (df['imdb_score'] > df['imdb_score'].mean()).astype(int)  # Binary label based on IMDb score

# Train the model
X = df[['num_critic_for_reviews', 'duration', 'num_voted_users', 'num_user_for_reviews', 'movie_facebook_likes', 'director_facebook_likes']]
y = df['success_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

def YT_Actor_Score():
    # Page title
    st.title("Movie Success Prediction from YouTube Trailer")

    # User input for YouTube link
    youtube_link = st.text_input("Enter a YouTube link")

    # Button to extract and display the video title
    if st.button("Calculate success"):
        if not youtube_link:
            st.error("Please enter a YouTube trailer link.")
        elif "youtube.com/watch?v=" not in youtube_link:
            st.error("Please enter a valid YouTube trailer link.")
        else:
            try:
                yt = YouTube(youtube_link)
                video_title = yt.title
                st.success(f"Movie: {video_title}")

                # Extract features for the movie from the YouTube trailer link
                features = [
                    yt.rating if yt.rating else 0,  # Use video rating if available, otherwise 0
                    yt.views,
                    yt.length,
                    yt.likes,
                    yt.dislikes,
                    yt.comments
                ]

                # Use the trained model to predict the success of the movie
                prediction = clf.predict([features])[0]
                prediction_text = "Successful" if prediction == 1 else "Not Successful"
                st.write(f"Prediction: {prediction_text}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Run the app
if __name__ == "__main__":
    YT_Actor_Score()
