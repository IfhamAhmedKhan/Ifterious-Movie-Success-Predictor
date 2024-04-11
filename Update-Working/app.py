import streamlit as st

# Define the movies and their details
movies = {
    "The Shawshank Redemption": {
        "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
        "options": ["The Shawshank Redemption", "Pulp Fiction", "The Godfather", "Forrest Gump"]
    },
    "The Godfather": {
        "description": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        "options": ["The Godfather", "The Shawshank Redemption", "Pulp Fiction", "Forrest Gump"]
    },
    "The Dark Knight": {
        "description": "When the menace known as The Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
        "options": ["The Dark Knight", "Inception", "Interstellar", "The Matrix"]
    },
    "Pulp Fiction": {
        "description": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
        "options": ["Pulp Fiction", "The Shawshank Redemption", "The Godfather", "Forrest Gump"]
    }
}

def movie_guessing_game():
    st.title("Movie Guessing Game")
    selected_movie = st.selectbox("Select a movie:", list(movies.keys()))

    if st.button("Check Answer"):
        correct_answer = movies[selected_movie]["options"][0]
        if selected_movie == correct_answer:
            st.write("Correct!")
        else:
            st.write(f"Incorrect. The correct answer was: {correct_answer}")

        st.write(f"Description: {movies[selected_movie]['description']}")

if __name__ == "__main__":
    movie_guessing_game()
