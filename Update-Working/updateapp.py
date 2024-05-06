import streamlit as st

def home():
    # Title
    st.title("Welcome to Ifterious Movie Success Predictor")

    # Container for movie cards
    st.write(
        """
        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <img src="https://via.placeholder.com/150" class="card-img-top" alt="...">
                    <div class="card-body">
                        <h5 class="card-title">Movie Title 1</h5>
                        <p class="card-text">Description of Movie 1.</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <img src="https://via.placeholder.com/150" class="card-img-top" alt="...">
                    <div class="card-body">
                        <h5 class="card-title">Movie Title 2</h5>
                        <p class="card-text">Description of Movie 2.</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <img src="https://via.placeholder.com/150" class="card-img-top" alt="...">
                    <div class="card-body">
                        <h5 class="card-title">Movie Title 3</h5>
                        <p class="card-text">Description of Movie 3.</p>
                    </div>
                </div>
            </div>
        </div>
        """
    , unsafe_allow_html=True)

# Display the home page
home()
