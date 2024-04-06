import streamlit as st
from streamlit_pages.streamlit_pages import MultiPage
from pytube import YouTube
import pandas as pd
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import torch
import plotly.express as px


# Define the dictionary of movies and their corresponding actors
movies_actors = {
    "Black Adam": ["Dwayne Johnson", "Aldis Hodge"],
    "Gemini Man": ["Will Smith"],
    "Pirates of the Caribbean": ["Johnny Depp", "Orlando Bloom"],
    "Mission: Impossible": ["Tom Cruise", "Jeremy Renner"],
    "Hobbs & Shaw": ["Dwayne Johnson", "Jason Statham"],
    "Avengers": ["Robert Downey Jr.", "Chris Evans"],
    "AMAZING SPIDER-MAN": ["Andrew Garfield", "Emma Stone"]
}

def home():
    st.header("Welcome to Ifterious Movie Success Predictor")

    # Display the image and text for the movie "Amazing Spiderman"
    st.image("movie_banner/amazing_spiderman.jpg", use_column_width=True)
    st.write("Movie: The Amazing Spider-Man")
    st.write("Description: The story of Spider-Man's origin, focusing on his high school years and his first encounter with the Lizard.")


def about():
    st.write("Welcome to about page")
    if st.button("Click about"):
        st.write("Welcome to About page")

def contact():
    st.write("Welcome to contact page")
    if st.button("Click Contact"):
        st.write("Welcome to contact page")

def YT_Actor_Score():
    # Page title
    st.title("Ifterious Movie Score Prediction")

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

                for movie, actors in movies_actors.items():
                    if movie in video_title:
                        st.info(f"Actors: {', '.join(actors)}")
                        break

                # Load the dataset
                df = pd.read_csv("movie_metadata.csv")

                # Data preprocessing
                df['gross'].fillna(df['gross'].mean(), inplace=True)
                df['budget'].fillna(df['budget'].mean(), inplace=True)
                df.dropna(inplace=True)
                df['main_genre'] = df['genres'].apply(lambda x: x.split('|')[0] if '|' in x else x)

                # Define a function to calculate actor score
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

                # Get the data for the actors
                actor_data = {}
                for actor in set(actors):
                    actor_data[actor] = df[df['actor_1_name'].str.contains(actor, case=False, na=False)]

                # Calculate the average actor score for each actor
                avg_actor_scores = {}
                for actor, data in actor_data.items():
                    avg_actor_scores[actor] = data['actor_score'].mean()

                # Calculate the combined average score
                combined_avg_score = sum(avg_actor_scores.values()) / len(avg_actor_scores)

                # Display individual movie scores for each actor
                for actor, data in actor_data.items():
                    st.write(f"\n{actor}'s Previous Movies and Scores:")
                    st.table(data[['movie_title', 'imdb_score', 'gross', 'budget', 'actor_score']])

                # Display the average actor scores
                for actor, avg_score in avg_actor_scores.items():
                    st.write(f"\nAverage Actor Score for {actor}: {avg_score:.2f}%")

                st.write(f"\nCombined Average Score: {combined_avg_score:.2f}%")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
                
# ---------------------------------

# Function to load actor embeddings and names
def load_actor_data(embeddings_directory):
    actor_embeddings = []
    actor_names = []

    for actor_folder in os.listdir(embeddings_directory):
        actor_folder_path = os.path.join(embeddings_directory, actor_folder)

        # Assuming each actor's folder contains files like "pins_ActorName_detected_face_*.npy"
        for embedding_file in os.listdir(actor_folder_path):
            embedding_file_path = os.path.join(actor_folder_path, embedding_file)

            # Load the embedding
            embedding = np.load(embedding_file_path)

            # Append the embedding and actor name
            actor_embeddings.append(embedding)
            actor_names.append(actor_folder)

    return np.array(actor_embeddings), np.array(actor_names)

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
    
    
def Actor_Score_FacialRecognition():
# Title
    st.title("Face Detection and Recognition")

    # File upload for first image
    uploaded_file1 = st.file_uploader("Upload the first image...", type=["jpg", "jpeg", "png"])

    # File upload for second image
    uploaded_file2 = st.file_uploader("Upload the second image...", type=["jpg", "jpeg", "png"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        # Display the uploaded images
        image1 = Image.open(uploaded_file1)
        image2 = Image.open(uploaded_file2)
        st.image([image1, image2], caption=["First Image", "Second Image"], use_column_width=True)

        # Resize the images to a smaller size (e.g., 170 x 240)
        resized_image1 = image1.resize((170, 240))
        resized_image2 = image2.resize((170, 240))

        # Convert resized images to numpy arrays
        image_np1 = np.array(resized_image1)
        image_np2 = np.array(resized_image2)

        # Initialize MTCNN for face detection
        mtcnn = MTCNN(keep_all=True)

        # Detect faces in the resized images
        boxes1, probs1 = mtcnn.detect(image_np1)
        boxes2, probs2 = mtcnn.detect(image_np2)

        if boxes1 is not None and boxes2 is not None:
            # Load actor embeddings and names
            embeddings_directory = 'working/extracted_embeddings'
            actor_embeddings, actor_names = load_actor_data(embeddings_directory)

            # Convert NumPy array to PyTorch tensor for the first image
            image_tensor1 = torch.tensor(image_np1.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

            # Convert NumPy array to PyTorch tensor for the second image
            image_tensor2 = torch.tensor(image_np2.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

            # Initialize InceptionResnetV1 for face embedding
            face_embedder = InceptionResnetV1(pretrained='vggface2').eval()

            # Calculate embeddings for the faces in the first image
            embedding1 = face_embedder(image_tensor1).detach().numpy()

            # Calculate embeddings for the faces in the second image
            embedding2 = face_embedder(image_tensor2).detach().numpy()

            # Calculate distances between the face embeddings and actor embeddings for the first image
            distances1 = euclidean_distances(embedding1, actor_embeddings)

            # Find the closest actor for the first image
            min_distance_index1 = np.argmin(distances1)
            closest_actor1 = actor_names[min_distance_index1]

            # Calculate distances between the face embeddings and actor embeddings for the second image
            distances2 = euclidean_distances(embedding2, actor_embeddings)

            # Find the closest actor for the second image
            min_distance_index2 = np.argmin(distances2)
            closest_actor2 = actor_names[min_distance_index2]

            # Remove the "pins_" prefix from the actor's names
            closest_actor1 = closest_actor1.replace('pins_', '')
            closest_actor2 = closest_actor2.replace('pins_', '')

            # Display the names of the closest actors
            st.write("Detected Actor in the First Image:", closest_actor1)
            st.write("Detected Actor in the Second Image:", closest_actor2)

            # Merge with the first script
            st.title("Movie Actor Success Predictor")

            df = pd.read_csv("movie_metadata.csv")

            # Calculate actor scores
            df['actor_score'] = df.apply(calculate_actor_score, axis=1)

            # User input for actor names
            actor_name_input1 = closest_actor1
            actor_name_input2 = closest_actor2

            # Calculate and display scores
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

                # Create a radar chart for visualization
                chart_data = pd.DataFrame({
                    'Actor': [actor_name_input1, actor_name_input2],
                    'Impact on Movie Success': [actor_avg_score1, actor_avg_score2]
                })
                fig = px.line_polar(chart_data, r='Impact on Movie Success', theta='Actor', line_close=True)
                st.write(fig)

            else:
                st.write(f"No data found for {actor_name_input1}. Please check the actor's name.")
                st.write(f"No data found for {actor_name_input2}. Please check the actor's name.")
        else:
            st.write("No faces detected in one or both images.")

# call app class object
app = MultiPage()
# Add pages
app.add_page("Home", home)
app.add_page("YouTube Movie Actor's Score System", YT_Actor_Score)
app.add_page("Actor Score with Images", Actor_Score_FacialRecognition)
app.add_page("About", about)
app.add_page("Contact", contact)
app.run()

