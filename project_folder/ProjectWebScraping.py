import streamlit as st
from pytube import YouTube
import cv2
import os
import pandas as pd

# Page title and description
st.title("Movie Success Predictor")
st.write("Enter a YouTube movie trailer link to get predictions on its potential success.")

# User input for YouTube link
youtube_link = st.text_input("Enter YouTube Link")

df = pd.read_csv("movie_metadata.csv")

# Display a preview of the dataset
# st.write("Dataset Preview:")
# st.write(df.head())

df['gross'].fillna(df['gross'].mean(), inplace=True)
df['budget'].fillna(df['budget'].mean(), inplace=True)
df.dropna(inplace=True)
df['main_genre'] = df['genres'].apply(lambda x: x.split('|')[0] if '|' in x else x)

# For working with classification models, I'm including a new column called "movie_status," which indicates whether a movie is successful (hit) or not (flop or average)
def getProject():
    return True

def getStatus(row):
    bgt = row['budget']
    grs = row['gross']

    if bgt * 2 <= grs:
        return 1
    return 0

df['movie_status'] = df[['budget', 'gross']].apply(getStatus, axis=1)  # 1 means HIT, 0 means FLOP

# Calculate a score based on IMDb scores and box office performance
def calculate_actor_score(row):
    imdb_weight = 0.5
    box_office_weight = 0.5

    imdb_score = row['imdb_score']
    box_office_performance = row['gross'] / row['budget'] if row['budget'] > 0 else 0

    # Normalize IMDb score to be between 0 and 1
    normalized_imdb_score = imdb_score / 10.0

    # Calculate the weighted average score
    weighted_score = (imdb_weight * normalized_imdb_score) + (box_office_weight * box_office_performance)

    # Ensure the final score is on a scale of 0 to 100%
    final_score = min(weighted_score * 100, 100)

    return final_score

# Apply the function to the DataFrame
df['actor_score'] = df.apply(calculate_actor_score, axis=1)

# Initialize frame_count to zero
frame_count = 0

# Movie name variable
movie_name = ""

# Download and frame extraction button
if st.button("Download and Extract Frames"):
    if not youtube_link:
        st.error("Please enter a YouTube link.")
    elif "youtube.com/watch?v=" not in youtube_link:
        st.error("Please enter a valid YouTube link.")
    else:
        try:
            placeholder_download_status = st.empty()  # Create an empty placeholder

            placeholder_download_status.info("Downloading the video...")

            # Create the "downloads" directory if it doesn't exist
            os.makedirs("downloads", exist_ok=True)

            # Download the YouTube video
            yt = YouTube(youtube_link)
            video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            video_path = os.path.join("downloads", yt.title + ".mp4")
            movie_name = yt.title  # Save the movie name
            video_stream.download(output_path="downloads/")

            # Update the download message
            placeholder_download_status.success("Download complete! Extracting frames...")

            # Create a "frames" directory for saving frames
            frames_directory = os.path.join("downloads", "frames", movie_name)
            os.makedirs(frames_directory, exist_ok=True)

            # Extract frames from the video with actor faces
            cap = cv2.VideoCapture(video_path)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces in the frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                if len(faces) > 0:
                    # Save frames as JPEG images with faces
                    frame_count += 1
                    frame_filename = os.path.join(frames_directory, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_filename, frame)

            cap.release()

            if frame_count > 0:
                st.success(f"Frame extraction completed! Frames with actor faces extracted: {frame_count}")
            else:
                st.warning("No frames with actor faces were extracted. Frame extraction may have failed.")

        except Exception as e:
            placeholder_download_status.error(f"An error occurred: {str(e)}")

# Progress bar for download and frame extraction
progress_bar = st.progress(0)
if frame_count > 0:
    for i in range(frame_count):
        progress_bar.progress((i + 1) / frame_count)

# Display the number of frames extracted with actor faces
st.write(f"Frames with actor faces extracted: {frame_count}")

# Display the movie name
st.write(f"Movie Name: {movie_name}")

# User input for actor names
actor_name_input1 = st.text_input("Enter the name of the first actor:")
actor_name_input2 = st.text_input("Enter the name of the second actor:")

# Button to calculate and display scores
if st.button("Calculate Actor Scores"):
    actor_data1 = df[df['actor_1_name'].str.contains(actor_name_input1, case=False, na=False)]
    actor_data2 = df[df['actor_1_name'].str.contains(actor_name_input2, case=False, na=False)]

    if not actor_data1.empty and not actor_data2.empty:
        # Display individual movie scores for actor 1
        st.write(f"{actor_name_input1}'s Previous Movies and Scores:")
        st.table(actor_data1[['movie_title', 'imdb_score', 'gross', 'budget', 'actor_score']])

        # Display individual movie scores for actor 2
        st.write(f"{actor_name_input2}'s Previous Movies and Scores:")
        st.table(actor_data2[['movie_title', 'imdb_score', 'gross', 'budget', 'actor_score']])

        # Calculate and display the overall average score for actor 1
        actor_avg_score1 = actor_data1['actor_score'].mean()
        st.write(f"\n{actor_name_input1}'s Overall Potential Impact on Movie's Success: {actor_avg_score1:.2f}%")

        # Calculate and display the overall average score for actor 2
        actor_avg_score2 = actor_data2['actor_score'].mean()
        st.write(f"\n{actor_name_input2}'s Overall Potential Impact on Movie's Success: {actor_avg_score2:.2f}%")

        # Calculate and display the overall average score for both actors combined
        combine_avg = ((actor_avg_score1 + actor_avg_score2) / 2)
        st.write(f"\nOverall Potential Impact of both actors on Movie's Success: {combine_avg:.2f}%")

    else:
        st.write(f"No data found for {actor_name_input1}. Please check the actor's name.")
        st.write(f"No data found for {actor_name_input2}. Please check the actor's name.")
