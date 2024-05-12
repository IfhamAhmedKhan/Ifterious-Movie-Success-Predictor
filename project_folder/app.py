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
import base64
import shutil

st.set_page_config(
    page_title="Ifterious Predictor",
    page_icon="Logo-white.png",  
    layout="centered"
)

@st.cache_data(show_spinner=False)  
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("flash.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.imgur.com/o2gErxv.png");
background-size: 100%;-
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;         /* Ensures the image covers the entire area */
    background-position: center; 
    background-repeat: no-repeat;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)



# Define the dictionary of movies and their corresponding actors
movies_actors = {
    "Black Adam": ["Dwayne Johnson", "Aldis Hodge"],
    "Gemini Man": ["Will Smith"],
    "Pirates of the Caribbean": ["Johnny Depp", "Orlando Bloom"],
    "Mission: Impossible": ["Tom Cruise", "Jeremy Renner"],
    "Hobbs & Shaw": ["Dwayne Johnson", "Jason Statham"],
    "Avengers": ["Robert Downey Jr.", "Chris Evans"],
    "AMAZING SPIDER-MAN": ["Andrew Garfield", "Emma Stone"],
    "Batman v Superman": ["Henry Cavil", "Ben Affleck"],
    "Suicide Squad": ["Ben Affleck","Margot Robbie"],
    "THE SUICIDE SQUAD": ["John Cena","Margot Robbie"],
    "BIRDS OF PREY": ["Margot Robbie","Rosie Perez"],
    "Inception" : ["Leonardo DiCaprio", "Tom Hardy"]
}

def home():
    st.markdown(
        """
        <head>
        <link rel="stylesheet" href="https://unpkg.com/boxicons@latest/css/boxicons.min.css">
          <link
            rel="stylesheet"
            href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"
          />
        </head>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div style="text-align: left; " ><img src="https://i.imgur.com/PvJhMlM.png" alt="Logo" width="300" height="300"></div>', unsafe_allow_html=True)

    st.markdown('<h1 class="animate__animated animate__fadeIn" style="text-align: left;">Welcome to Ifterious Movie Success Predictor</h1>', unsafe_allow_html=True)

    st.markdown("""
<style>
p {
    font-size: 24px;
    text-align: left;
    line-height: 1.5;
    margin-bottom: 40px;
}
</style>
<p>Explore the world of movies with our<br> interactive app. Analyze movie data,<br> predict actor scores, and dive into<br> fascinating insights.</p><br><br><br><br><br><br><br>
""", unsafe_allow_html=True)


    st.write(
        """
        <style>
        .center-content {
            text-align: center;
        }
        .stars {
            color: gold;
        }
        </style>
        
        <div class="row center-content">
        <div class="col-md-4" style= "border: 2px solid black; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://i.imgur.com/PjlS5r0.jpeg" class="card-img-top" alt="..."  width="250" height="380">
                    <div class="card-body">
                        <h5 class="card-title">Movie Title: Dune: Part One</h5>
                        <p class="card-text">Description: A noble family becomes embroiled in a war for control over the galaxy's most valuable asset while its heir becomes troubled by visions of a dark future.</p>
                        <div class="stars">
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class="bx bx-star"></i>
        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4" style= "border: 2px solid black; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://m.media-amazon.com/images/M/MV5BYTM3ZTllNzItNTNmOS00NzJiLTg1MWMtMjMxNDc0NmJhODU5XkEyXkFqcGdeQXVyODE5NzE3OTE@._V1_.jpg" class="card-img-top" alt="..."  width="250" height="380">
                    <div class="card-body">
                        <h5 class="card-title">Movie Title: Kingsman: The Secret Service</h5>
                        <p class="card-text">Description: A spy organisation recruits a promising street kid into the agency's training program, while a global threat emerges from a twisted tech genius.</p>
                        <div class="stars">
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star-half'></i>
        </div>
                    </div>
                </div>
            </div>
        <div class="col-md-4" style= "border: 2px solid black; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://m.media-amazon.com/images/M/MV5BMTc5MDE2ODcwNV5BMl5BanBnXkFtZTgwMzI2NzQ2NzM@._V1_.jpg" class="card-img-top" alt="..." width="250" height="380">
                    <div class="card-body">
                        <h5 class="card-title">Movie Title: Avengers: Endgame</h5>
                        <p class="card-text">Description: After the devastating events of Avengers: Infinity War (2018), the universe is in ruins. With the help of remaining allies, the Avengers assemble once more in order to reverse Thanos' actions and restore balance to the universe.</p>
                        <div class="stars">
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star-half'></i>
        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4" style= "border: 2px solid black; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://upload.wikimedia.org/wikipedia/en/e/e0/The_Amazing_Spider-Man_%28film%29_poster.jpg" class="card-img-top" alt="...">
                    <div class="card-body">
                        <h5 class="card-title" text-align="center">Movie Title: The Amazing Spider-Man</h5>
                        <p class="card-text" id="movie-description">Description: Peter Parker, a shy and brilliant high school student, gains extraordinary spider-like abilities after a fateful bite. As he navigates adolescence, Peter must learn to use his newfound powers for good while facing personal challenges and battling dangerous villains that threaten his city..</p>
                        <div class="stars">
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star-half'></i>
        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4" style= "border: 2px solid black; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://upload.wikimedia.org/wikipedia/en/a/a9/Black_Adam_%28film%29_poster.jpg" class="card-img-top" alt="...">
                    <div class="card-body">
                        <h5 class="card-title">Movie Title: Black Adam</h5>
                        <p class="card-text">Description: Kahndaq, a land ravaged by tyranny. Teth-Adam, a man desperate to save his family, seeks the power of champions. Yet, the magic corrupts, twisting him into Black Adam. Centuries later, archaeologists unleash his fury. Now, Black Adam must confront his past and choose: remain a slave to rage or become the hero Kahndaq needs.</p>
                        <div class="stars">
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class="bx bx-star"></i>
        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4" style= "border: 2px solid black; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://m.media-amazon.com/images/M/MV5BOTIzYmUyMmEtMWQzNC00YzExLTk3MzYtZTUzYjMyMmRiYzIwXkEyXkFqcGdeQXVyMDM2NDM2MQ@@._V1_FMjpg_UX1000_.jpg" class="card-img-top" alt="..." width="250" height="380">
                    <div class="card-body">
                        <h5 class="card-title">Movie Title: Fast & Furious Presents: Hobbs & Shaw</h5>
                        <p class="card-text">Description: Worlds collide when DSS agent Luke Hobbs and rogue assassin Deckard Shaw are forced to team up against a cyber-genetically enhanced threat. From Los Angeles to London, these unlikely allies ignite a trail of high-octane action and witty banter.  But can they put their differences aside to save the world?</p>
                        <div class="stars">
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star-half'></i>
        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    , unsafe_allow_html=True)

def about():
    st.write(
    """
    <style>
    *{
    color: #FFFACD;
    }

    .center-content {
        text-align: center;
    }

    .team-row {
        display: flex;
        justify-content: center; /* Center the cards horizontally */
        align-items: flex-start; /* Align items at the start of the cross axis (top) */
        flex-wrap: wrap;
        margin-top: 20px; /* Add margin at the top for spacing */
    }

    .card {
        width: 250px;
        margin: 10px;
        background-color:black;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    a {
        text-decoration: none;
    }
    .card-body {
        padding: 15px; /* Add padding inside the card body for spacing */
    }

    .card-title {
        text-align: center; /* Center the card title */
    }


    </style>

    <div class="center-content"> 
        <h1>About Ifterious Movie Success Predictor</h1>
        <p>Ifterious Movie Success Predictor is a project aimed at predicting the success of movies based on various factors, including actor performance. Using machine learning and data analysis techniques, we strive to provide insights into what makes a movie successful in today's competitive entertainment industry.</p>
        <h1>Our Mission</h1>
        <p>Our mission is to help filmmakers, producers, and movie enthusiasts understand the dynamics behind a movie's success. By analyzing data such as box office performance, IMDb scores, and actor impact, we aim to empower decision-makers to make informed choices in their movie production and selection processes.</p>
        <div class="team-row">  
            <div class="card">
                <img src="https://ifterious-tech.netlify.app/main_web/bq-final-project-images--main/IMG_20211027_234719.jpg" class="card-img-top" alt="..." width="250" height="380">
                <div class="card-body">
                    <h5 class="card-title" text-align="center">Name: Ifham Ahmed Khan</h5>
                    <p class="card-text">Email: ifham.khan105@gmail.com</p>
                    <p class="card-text">Phone: +92 316 1611907</p>
                    <p class="card-text">Github: <a href="https://github.com/IfhamAhmedKhan">IfhamAhmedKhan</a></p>
                    <p class="card-text">Linkedin: <a href="https://www.linkedin.com/in/ifham-khan-479332278/">Ifham Khan</a></p>
                </div>
            </div>
            <div class="card">
                <img src="https://i.imgur.com/PEz6M2N.jpeg" class="card-img-top" alt="..." width="250" height="380">
                <div class="card-body" >
                    <h5 class="card-title">Name: Asad Iqbal</h5><br>
                    <p class="card-text">Email: asad.iqbal5165@gmail.com</p>
                    <p class="card-text">Phone: +92 340 2671795</p>
                    <p class="card-text">Github: <a href="https://github.com/AsadIqbal5165">AsadIqbal5165</a></p>
                    <p class="card-text">Linkedin: <a href="https://www.linkedin.com/in/asad-iqbal-699803234/">Asad Iqbal</a></p>
                </div>
            </div>
            <div class="card">
                <img src="https://cdn.vectorstock.com/i/500p/50/18/portrait-photo-icon-vector-31995018.jpg" class="card-img-top" alt="..." width="250" height="380">
                <div class="card-body">
                    <h5 class="card-title">Name: Abdul Aziz</h5>
                    <p class="card-text">Email: abdulazizk811@gmail.com</p>
                    <p class="card-text">Phone: +92 337 8057564</p>
                    <p class="card-text">Github: <a href="https://github.com/abdulazizk2">abdulazizk2</a></p>
                    <p class="card-text">Linkedin: <a href="">Abdul Aziz</a></p>
                </div>
            </div>
        </div>
    </div>
    <div class="footer">
        
        
    </div>
    """
    , unsafe_allow_html=True)




    

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
                # Create the "downloads" directory if it doesn't exist
                os.makedirs("downloads", exist_ok=True)

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

                # Delete everything inside the "downloads" folder
                folder = 'downloads'
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        st.error(f"An error occurred while deleting {file_path}: {e}")

                #st.success("All files in 'downloads' folder deleted successfully!")

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
app.run()



