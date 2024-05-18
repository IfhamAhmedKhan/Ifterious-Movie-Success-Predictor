import streamlit as st
from streamlit_pages.streamlit_pages import MultiPage
from pytube import YouTube
import pandas as pd
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import plotly.express as px
import base64
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Ifterious Predictor",
    page_icon="Logo-black.png",  
    layout="centered"
)

@st.cache_data(show_spinner=False)  
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("starwars.jpg")

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

@st.cache_data
def load_data():
    df = pd.read_csv("movie_metadata.csv")
    df['gross'].fillna(df['gross'].mean(), inplace=True)
    df['budget'].fillna(df['budget'].mean(), inplace=True)
    
    # Count rows before dropping NA
    initial_rows = df.shape[0]
    
    df.dropna(inplace=True)
    
    # Count rows after dropping NA
    final_rows = df.shape[0]
    
    # Log the number of rows dropped
    st.write(f"Rows before dropping NA: {initial_rows}")
    st.write(f"Rows after dropping NA: {final_rows}")
    
    df['main_genre'] = df['genres'].apply(lambda x: x.split('|')[0] if '|' in x else x)
    df['success_label'] = (df['imdb_score'] > df['imdb_score'].mean()).astype(int)
    return df

@st.cache_data
def train_model(df):
    X = df[['num_critic_for_reviews', 'duration', 'num_voted_users', 'num_user_for_reviews', 'movie_facebook_likes', 'director_facebook_likes']]
    y = df['success_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #st.write(f"Model Accuracy: {accuracy:.2f}")
    return clf

df = load_data()
clf = train_model(df)

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

    st.markdown('<div style="text-align: left; " ><img src="https://i.imgur.com/PvJhMlM.png" alt="Logo" width="300" height="300" class="animate__animated animate__fadeInLeft"></div>', unsafe_allow_html=True)

    st.markdown('<h1 class="animate__animated animate__fadeIn" style="text-align: left;">Welcome to Ifterious Movie Success Predictor</h1>', unsafe_allow_html=True)

    st.markdown("""
<style>
p {
    font-size: 24px;
    text-align: left;
    line-height: 1.5;
    margin-bottom: 40px;
}
 @keyframes typing {
  from { width: 0; }
  to { width: 100%; }
}

@keyframes blink-caret {
  from, to { border-color: transparent; }
  50% { border-color: black; }
}

.typing {
  font-family: Arial, sans-serif;
  font-size: 24px;
  line-height: 1.5;
  color: #FFFACD;
  white-space: nowrap;
  overflow: hidden;
  border-right: 3px solid black; /* Cursor */
  animation: 
    typing 4s steps(40, end), 
    blink-caret 0.75s step-end infinite;
  display: inline-block;
  max-width: 100%; /* Ensure the text wraps correctly */
}


</style>
<p class="typing">Explore the world of movies with our project. <br>Analyze movie data, predict actor scores. <br>Let's dive into fascinating insights.</p><br><br><br><br><br><br><br><br><br><br><br><br>
""", unsafe_allow_html=True)


    st.markdown("""
<head>
<style>
.center-content {
    text-align: center;
}
.stars {
    color: gold;
}
.card-title {
    font-size: 30px;
}
.card-text {
    font-family: 'Great Vibes', cursive;
}
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css" integrity="sha512-SnH5WK+bZxgPHs44uWIX+LLJAJ9/2PkPKZ5QiAj6Ta86w+fsb2TkcmfRyVX3pBnMFcV7oQPJkl9QevSCWr3W6A==" crossorigin="anonymous" referrerpolicy="no-referrer" />
</head>
<div class="row center-content">
<div class="col-md-4" style= "border: 4px solid #131bbd; border-radius: 25px; padding: 5px;">
        <div class="card" >
            <img src="https://shop.legendary.com/cdn/shop/files/Dune-min.png" class="card-img-top" alt="..."  width="690" height="380" style="border-radius: 20px;">
            <div class="card-body">
                <h5 class="card-title" style="border: 2px solid black; border-radius: 25px; padding: 5px;margin: 5px; background-color: #168991">Movie Title: Dune: Part One</h5>
                <p class="card-text"  style="text-align:center;">Description: A noble family becomes embroiled in a war for control over the galaxy's most valuable asset while its heir becomes troubled by visions of a dark future.</p>
                <p style="text-align: center; margin-bottom: 0;">Watch trailer</p>
                <a href="https://www.youtube.com/watch?v=n9xhJrPXop4"><i class="fab fa-youtube fa-4x" style="color: #ff0000;"></i></a>
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
                <br><br>
            <div class="col-md-4" style= "border: 4px solid #131bbd; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://images.thedirect.com/media/article_full/newpos_QB7hEyO.jpg" class="card-img-top" alt="..." width="690" height="380" style="border-radius: 20px;">
                    <div class="card-body">
                        <h5 class="card-title" style="border: 2px solid black; border-radius: 25px; padding: 5px;margin: 5px; background-color: #168991">Movie Title: Zack Snyder's Justice League</h5>
                        <p class="card-text" style="text-align:center;">Description: Determined to ensure that Superman's ultimate sacrifice wasn't in vain, Bruce Wayne recruits a team of metahumans to protect the world from an approaching threat of catastrophic proportions.</p>
                        <p style="text-align: center; margin-bottom: 0;">Watch trailer</p>
                <a href="https://www.youtube.com/watch?v=ui37YKQ9AC4"><i class="fab fa-youtube fa-4x" style="color: #ff0000;"></i></a>
                <div class="stars">
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bxs-star'></i>
            <i class='bx bx-star'></i>
        </div>
                    </div>
                </div>
            </div>
                <br><br>
                <div class="col-md-4" style= "border: 4px solid #131bbd; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://images8.alphacoders.com/112/1121819.jpg" class="card-img-top" alt="..."   width="690" height="380" style="border-radius: 20px;">
                    <div class="card-body">
                        <h5 class="card-title" style="border: 2px solid black; border-radius: 25px; padding: 5px;margin: 5px; background-color: #168991">Movie Title: Kingsman: The Secret Service</h5>
                        <p class="card-text" style="text-align:center;">Description: A spy organisation recruits a promising street kid into the agency's training program, while a global threat emerges from a twisted tech genius.</p>
                        <p style="text-align: center; margin-bottom: 0;">Watch trailer</p>
                <a href="https://www.youtube.com/watch?v=m4NCribDx4U"><i class="fab fa-youtube fa-4x" style="color: #ff0000;"></i></a>
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
                <br><br>
        <div class="col-md-4" style= "border: 4px solid #131bbd; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://images2.alphacoders.com/111/1119554.jpg" class="card-img-top" alt="..." width="690" height="380" style="border-radius: 20px;">
                    <div class="card-body">
                        <h5 class="card-title" style="border: 2px solid black; border-radius: 25px; padding: 5px;margin: 5px; background-color: #168991">Movie Title: Avengers: Endgame</h5>
                        <p class="card-text" style="text-align:center;">Description: After the devastating events of Avengers: Infinity War (2018), the universe is in ruins. With the help of remaining allies, the Avengers assemble once more in order to reverse Thanos' actions and restore balance to the universe.</p>
                <p style="text-align: center; margin-bottom: 0;">Watch trailer</p>
                <a href="https://www.youtube.com/watch?v=TcMBFSGVi1c&t"><i class="fab fa-youtube fa-4x" style="color: #ff0000;"></i></a>
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
                <br><br>
            <div class="col-md-4" style= "border: 4px solid #131bbd; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://cinemasiren.com/wp-content/uploads/2014/05/AmazingSpiderMan2Banner.jpg" class="card-img-top" alt="..."  width="690" height="380" style="border-radius: 20px;">
                    <div class="card-body">
                        <h5 class="card-title" text-align="center" style="border: 2px solid black; border-radius: 25px; padding: 5px;margin: 5px; background-color: #168991">Movie Title: The Amazing Spider-Man 2</h5>
                        <p class="card-text" id="movie-description"  style="text-align:center;">Description: Peter Parker, a shy and brilliant high school student, gains extraordinary spider-like abilities after a fateful bite. As he navigates adolescence, Peter must learn to use his newfound powers for good while facing personal challenges and battling dangerous villains that threaten his city..</p>
                <p style="text-align: center; margin-bottom: 0;">Watch trailer</p>
                <a href="https://www.youtube.com/watch?v=nbp3Ra3Yp74"><i class="fab fa-youtube fa-4x" style="color: #ff0000;"></i></a>
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
                <br><br>
            <div class="col-md-4" style= "border: 4px solid #131bbd; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://images2.alphacoders.com/130/1300734.jpg" class="card-img-top" alt="..."  width="690" height="380" style="border-radius: 20px;">
                    <div class="card-body">
                        <h5 class="card-title" style="border: 2px solid black; border-radius: 25px; padding: 5px;margin: 5px; background-color: #168991">Movie Title: Black Adam</h5>
                        <p class="card-text" style="text-align:center;">Description: Kahndaq, a land ravaged by tyranny. Teth-Adam, a man desperate to save his family, seeks the power of champions. Yet, the magic corrupts, twisting him into Black Adam. Centuries later, archaeologists unleash his fury. Now, Black Adam must confront his past and choose: remain a slave to rage or become the hero Kahndaq needs.</p>
                <p style="text-align: center; margin-bottom: 0;">Watch trailer</p>
                <a href="https://www.youtube.com/watch?v=X0tOpBuYasI&t"><i class="fab fa-youtube fa-4x" style="color: #ff0000;"></i></a>
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
                <br><br>
            <div class="col-md-4" style= "border: 4px solid #131bbd; border-radius: 25px; padding: 5px;">
                <div class="card">
                    <img src="https://images.alphacoders.com/130/thumb-1920-1300729.jpg" class="card-img-top" alt="..." width="690" height="380" style="border-radius: 20px;">
                    <div class="card-body">
                        <h5 class="card-title"  style="border: 2px solid black; border-radius: 25px; padding: 5px;margin: 5px; background-color: #168991">Movie Title: Fast & Furious Presents: Hobbs & Shaw</h5>
                        <p class="card-text" style="text-align:center;">Description: Worlds collide when DSS agent Luke Hobbs and rogue assassin Deckard Shaw are forced to team up against a cyber-genetically enhanced threat. From Los Angeles to London, these unlikely allies ignite a trail of high-octane action and witty banter.  But can they put their differences aside to save the world?</p>
                <p style="text-align: center; margin-bottom: 0;">Watch trailer</p>
                <a href="https://www.youtube.com/watch?v=HZ7PAyCDwEg&t"><i class="fab fa-youtube fa-4x" style="color: #ff0000;"></i></a>
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
    h1,p,h2 {
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
        background-color: black; /* Set background color to black */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-radius: 15px;
        overflow: hidden;
    }

    .card img {
        border-top-left-radius: 15px;
        border-top-right-radius: 15px;
    }

    .card-body {
        padding: 15px; /* Add padding inside the card body for spacing */
    }

    .card-title {
        text-align: center; /* Center the card title */
        font-size: 1.2em;
        color: #FFFACD;
    }

    .card-text {
        font-size: 1em;
        color: #FFFACD;
        text-align: center;
    }

    .card a {
        color: #FFA07A; /* Soft coral color for links */
        text-decoration: none
    }

    .card a:hover {
        color: #FF6347; /* Tomato color for links on hover */
    }

    .footer {
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        background-color: #0E1117;
        color: #FFFACD;
    }

    </style>

    <div class="center-content"> 
        <h1 class="textHover">About Ifterious Movie Success Predictor</h1>
        <p>Ifterious Movie Success Predictor is a project aimed at predicting the success of movies based on various factors, including actor performance. Using machine learning and data analysis techniques, we strive to provide insights into what makes a movie successful in today's competitive entertainment industry.</p>
        <h1>Our Mission</h1>
        <p>Our mission is to help filmmakers, producers, and movie enthusiasts understand the dynamics behind a movie's success. By analyzing data such as box office performance, IMDb scores, and actor impact, we aim to empower decision-makers to make informed choices in their movie production and selection processes.</p>
        <h1>Features</h1>
        <h2>Movie Success Prediction</h2>
        <p>Our application predicts the success of movies based on historical data and various performance metrics. We use machine learning algorithms to analyze and predict the potential success of movies, helping stakeholders make data-driven decisions.</p>
        <h2>Actor Score Calculation</h2>
        <p>We calculate scores for actors based on their past performances and other relevant factors. This helps in understanding the impact of an actor's presence in a movie on its overall success.</p>
        <h2>Facial Recognition</h2>
        <p>Our facial recognition feature allows users to upload images of actors, and our system will identify them and provide their performance scores. This feature is powered by advanced deep learning models for accurate and efficient recognition.</p>
        <h2>Story Generation</h2>
        <p>Using GPT-2, we offer a story generation feature where users can input a story idea, and our system will generate a complete story based on the given idea and selected genre. This feature leverages state-of-the-art natural language processing techniques to create engaging and creative stories.</p>
        <div class="team-row">  
            <div class="card">
                <img src="https://ifterious-tech.netlify.app/main_web/bq-final-project-images--main/IMG_20211027_234719.jpg" class="card-img-top" alt="Ifham Ahmed Khan" width="250" height="380">
                <div class="card-body">
                    <h5 class="card-title">Ifham Ahmed Khan</h5>
                    <p class="card-text">Email: ifham.khan105@gmail.com</p>
                    <p class="card-text">Phone: +92 316 1611907</p>
                    <p class="card-text">Github: <a href="https://github.com/IfhamAhmedKhan">IfhamAhmedKhan</a></p>
                    <p class="card-text">Linkedin: <a href="https://www.linkedin.com/in/ifham-khan-479332278/">Ifham Khan</a></p>
                </div>
            </div>
            <div class="card">
                <img src="https://i.imgur.com/PEz6M2N.jpeg" class="card-img-top" alt="Asad Iqbal" width="250" height="380">
                <div class="card-body">
                    <h5 class="card-title">Asad Iqbal</h5>
                    <p class="card-text">Email: asad.iqbal5165@gmail.com</p>
                    <p class="card-text">Phone: +92 340 2671795</p>
                    <p class="card-text">Github: <a href="https://github.com/AsadIqbal5165">AsadIqbal5165</a></p>
                    <p class="card-text">Linkedin: <a href="https://www.linkedin.com/in/asad-iqbal-699803234/">Asad Iqbal</a></p>
                </div>
            </div>
            <div class="card">
                <img src="https://cdn.vectorstock.com/i/500p/50/18/portrait-photo-icon-vector-31995018.jpg" class="card-img-top" alt="Abdul Aziz" width="250" height="380">
                <div class="card-body">
                    <h5 class="card-title">Abdul Aziz</h5>
                    <p class="card-text">Email: abdulazizk811@gmail.com</p>
                    <p class="card-text">Phone: +92 337 8057564</p>
                    <p class="card-text">Github: <a href="https://github.com/abdulazizk2">abdulazizk2</a></p>
                    <p class="card-text">Linkedin: <a href="">Abdul Aziz</a></p>
                </div>
            </div>
        </div>
    </div>
    <div class="footer" style="">
        <p>&copy; 2024 Ifterious Movie Success Predictor. All rights reserved.</p>
    </div>
    """
    , unsafe_allow_html=True)


def YT_Actor_Score():
    st.title("Ifterious Movie Score Prediction")
    youtube_link = st.text_input("Enter a YouTube link")

    if st.button("Calculate success"):
        if not youtube_link:
            st.error("Please enter a YouTube trailer link.")
        elif "youtube.com/watch?v=" not in youtube_link:
            st.error("Please enter a valid YouTube trailer link.")
        else:
            try:
                os.makedirs("downloads", exist_ok=True)
                yt = YouTube(youtube_link)
                video_title = yt.title
                st.success(f"Movie: {video_title}")

                df['actor_score'] = df.apply(calculate_actor_score, axis=1)
                movies_actors = pd.read_csv('movies_actors.csv')
                movies_actors_dict = {row['Movie']: row['Actors'].split(',') for _, row in movies_actors.iterrows()}

                actors = next((actors for movie, actors in movies_actors_dict.items() if movie in video_title), None)
                if not actors:
                    movie_row = df[df['movie_title'].str.contains(video_title, case=False, na=False)]
                    if not movie_row.empty:
                        actors = [movie_row.iloc[0][col] for col in ['actor_1_name', 'actor_2_name', 'actor_3_name']]

                if actors:
                    actor_data = {actor: df[
                        (df['actor_1_name'].str.contains(actor, case=False, na=False)) |
                        (df['actor_2_name'].str.contains(actor, case=False, na=False)) |
                        (df['actor_3_name'].str.contains(actor, case=False, na=False))
                    ] for actor in set(actors)}

                    avg_actor_scores = {actor: data['actor_score'].mean() for actor, data in actor_data.items()}
                    combined_avg_score = sum(avg_actor_scores.values()) / len(avg_actor_scores)

                    for actor, data in actor_data.items():
                        st.write(f"\n{actor}'s Previous Movies and Scores:")
                        st.table(data[['movie_title', 'imdb_score', 'gross', 'budget', 'actor_score']])
                        st.subheader(f"Individual Movie Scores for {actor}")
                        fig_bar = px.bar(data, x='movie_title', y='actor_score', title=f"{actor}'s Movie Scores")
                        st.write(fig_bar)

                    for actor, avg_score in avg_actor_scores.items():
                        st.write(f"\nAverage Actor Score for {actor}: {avg_score:.2f}%")
                    st.write(f"\nCombined Average Score: {combined_avg_score:.2f}%")

                    st.subheader("Overall Potential Impact of Actors")
                    radar_data = pd.DataFrame({
                        'Actor': list(avg_actor_scores.keys()),
                        'Impact on Movie Success': list(avg_actor_scores.values())
                    })
                    fig_radar = px.line_polar(radar_data, r='Impact on Movie Success', theta='Actor', line_close=True)
                    st.write(fig_radar)

                    st.subheader("Distribution of Actor Scores")
                    all_actor_scores = [data['actor_score'] for data in actor_data.values()]
                    box_data = pd.DataFrame({
                        'Actor': [actor for actor in actor_data for _ in range(len(actor_data[actor]))],
                        'Actor Score': np.concatenate(all_actor_scores)
                    })
                    fig_box = px.box(box_data, x='Actor', y='Actor Score', title="Actor Score Distribution")
                    st.write(fig_box)

                    st.subheader("Feature Correlation Heatmap")
                    correlation = df[['imdb_score', 'gross', 'budget', 'actor_score']].corr()
                    fig_heatmap = px.imshow(correlation, text_auto=True, title="Feature Correlation Heatmap")
                    st.write(fig_heatmap)

                    shutil.rmtree('downloads', ignore_errors=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# ---------------------------------

def calculate_actor_score(row):
    imdb_weight = 0.5
    box_office_weight = 0.5
    imdb_score = row['imdb_score']
    box_office_performance = row['gross'] / row['budget'] if row['budget'] > 0 else 0
    normalized_imdb_score = imdb_score / 10.0
    weighted_score = (imdb_weight * normalized_imdb_score) + (box_office_weight * box_office_performance)
    return min(weighted_score * 100, 100)

def load_actor_data(embeddings_directory):
    actor_embeddings = []
    actor_names = []

    for actor_folder in os.listdir(embeddings_directory):
        actor_folder_path = os.path.join(embeddings_directory, actor_folder)
        for embedding_file in os.listdir(actor_folder_path):
            embedding_file_path = os.path.join(actor_folder_path, embedding_file)
            embedding = np.load(embedding_file_path)
            actor_embeddings.append(embedding)
            actor_names.append(actor_folder)

    return np.array(actor_embeddings), np.array(actor_names)

def Actor_Score_FacialRecognition():
    st.title("Face Detection and Recognition")
    uploaded_file1 = st.file_uploader("Upload the first image...", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Upload the second image...", type=["jpg", "jpeg", "png"])

    if uploaded_file1 and uploaded_file2:
        image1 = Image.open(uploaded_file1)
        image2 = Image.open(uploaded_file2)
        st.image([image1, image2], caption=["First Image", "Second Image"], use_column_width=True)

        resized_image1 = image1.resize((170, 240))
        resized_image2 = image2.resize((170, 240))
        image_np1 = np.array(resized_image1)
        image_np2 = np.array(resized_image2)

        mtcnn = MTCNN(keep_all=True)
        boxes1, _ = mtcnn.detect(image_np1)
        boxes2, _ = mtcnn.detect(image_np2)

        if boxes1 is not None and boxes2 is not None:
            embeddings_directory = 'working/extracted_embeddings'
            actor_embeddings, actor_names = load_actor_data(embeddings_directory)

            image_tensor1 = torch.tensor(image_np1.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            image_tensor2 = torch.tensor(image_np2.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

            face_embedder = InceptionResnetV1(pretrained='vggface2').eval()
            embedding1 = face_embedder(image_tensor1).detach().numpy()
            embedding2 = face_embedder(image_tensor2).detach().numpy()

            distances1 = euclidean_distances(embedding1, actor_embeddings)
            closest_actor1 = actor_names[np.argmin(distances1)].replace('pins_', '')

            distances2 = euclidean_distances(embedding2, actor_embeddings)
            closest_actor2 = actor_names[np.argmin(distances2)].replace('pins_', '')

            st.write("Detected Actor in the First Image:", closest_actor1)
            st.write("Detected Actor in the Second Image:", closest_actor2)

            df['actor_score'] = df.apply(calculate_actor_score, axis=1)

            actor_data1 = df[
                df['actor_1_name'].str.contains(closest_actor1, case=False, na=False) |
                df['actor_2_name'].str.contains(closest_actor1, case=False, na=False) |
                df['actor_3_name'].str.contains(closest_actor1, case=False, na=False)
            ]

            actor_data2 = df[
                df['actor_1_name'].str.contains(closest_actor2, case=False, na=False) |
                df['actor_2_name'].str.contains(closest_actor2, case=False, na=False) |
                df['actor_3_name'].str.contains(closest_actor2, case=False, na=False)
            ]

            if not actor_data1.empty and not actor_data2.empty:
                st.write(f"{closest_actor1}'s Previous Movies and Scores:")
                st.table(actor_data1[['movie_title', 'imdb_score', 'gross', 'budget', 'actor_score']])
                st.subheader(f"Bar Chart for {closest_actor1}'s Movie Scores")
                fig1 = px.bar(actor_data1, x='movie_title', y='actor_score', title=f"{closest_actor1}'s Movie Scores")
                st.plotly_chart(fig1)

                st.write(f"{closest_actor2}'s Previous Movies and Scores:")
                st.table(actor_data2[['movie_title', 'imdb_score', 'gross', 'budget', 'actor_score']])
                st.subheader(f"Bar Chart for {closest_actor2}'s Movie Scores")
                fig2 = px.bar(actor_data2, x='movie_title', y='actor_score', title=f"{closest_actor2}'s Movie Scores")
                st.plotly_chart(fig2)

                actor_avg_score1 = actor_data1['actor_score'].mean()
                st.write(f"\n{closest_actor1}'s Overall Potential Impact on Movie's Success: {actor_avg_score1:.2f}%")

                actor_avg_score2 = actor_data2['actor_score'].mean()
                st.write(f"\n{closest_actor2}'s Overall Potential Impact on Movie's Success: {actor_avg_score2:.2f}%")

                combine_avg = ((actor_avg_score1 + actor_avg_score2) / 2)
                st.write(f"\nOverall Potential Impact of both actors on Movie's Success: {combine_avg:.2f}%")

                st.subheader("Radar Chart for Overall Impact")
                chart_data = pd.DataFrame({
                    'Actor': [closest_actor1, closest_actor2],
                    'Impact on Movie Success': [actor_avg_score1, actor_avg_score2]
                })
                fig3 = px.line_polar(chart_data, r='Impact on Movie Success', theta='Actor', line_close=True)
                st.plotly_chart(fig3)

                st.subheader("Box Plot for Score Distribution")
                combined_actor_data = pd.concat([actor_data1, actor_data2])
                fig4 = px.box(combined_actor_data, x='actor_1_name', y='actor_score', title='Score Distribution')
                st.plotly_chart(fig4)

                st.subheader("Heatmap for Feature Correlation")
                correlation = df[['imdb_score', 'gross', 'budget', 'actor_score']].corr()
                fig5 = px.imshow(correlation, text_auto=True, title='Feature Correlation')
                st.plotly_chart(fig5)

            else:
                if actor_data1.empty:
                    st.write(f"No data found for {closest_actor1}. Please check the actor's name.")
                if actor_data2.empty:
                    st.write(f"No data found for {closest_actor2}. Please check the actor's name.")
        else:
            st.write("No faces detected in one or both images.")

@st.cache_resource
def load_model(model_name="gpt2"):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model()

# Function to generate a story using GPT-2
def generate_story(idea, max_length=300, temperature=1.0):
    input_ids = tokenizer.encode(idea, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, no_repeat_ngram_size=2, early_stopping=True)
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

# Story Generator Page
def Story_Gen():
    st.title("Ifterious Story Generator")

    # Story genre selection
    genres = ["Fantasy", "Science Fiction", "Romance", "Mystery"]
    selected_genre = st.selectbox("Choose a genre", genres)

    # Genre-specific prompts
    genre_prompts = {
        "Fantasy": "Once upon a time in a magical kingdom,",
        "Science Fiction": "In a distant future,",
        "Romance": "On a warm summer day,",
        "Mystery": "In a small town, a detective discovered,"
    }

    # Input form for the story idea with default genre prompt
    idea = st.text_input("Enter your story idea:", genre_prompts[selected_genre])

    # Customizable output settings
    max_length = st.slider("Story Length", min_value=50, max_value=500, value=300)
    temperature = st.slider("Creativity", min_value=0.7, max_value=1.5, value=1.0)

    # Button to generate story
    if st.button("Generate Story"):
        if idea:
            with st.spinner('Generating story...'):
                story = generate_story(idea, max_length, temperature)
            st.subheader("Generated Story:")
            st.write(story)
        else:
            st.error("Please enter a story idea.")

# call app class object
app = MultiPage()
# Add pages
app.add_page("Home", home)
app.add_page("YouTube Movie Actor's Score System", YT_Actor_Score)
app.add_page("Actor Score with Images", Actor_Score_FacialRecognition)
app.add_page("Story generator", Story_Gen)
app.add_page("About", about)
app.run()