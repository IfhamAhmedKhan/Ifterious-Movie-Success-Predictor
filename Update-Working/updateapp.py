import streamlit as st
import base64

# Function to load and display the background video
def add_bg_video(video_path):
    """Adds a background video to the Streamlit app."""
    video_file = open(video_path, "rb")
    video_bytes = video_file.read()
    data_url = base64.b64encode(video_bytes).decode("utf-8")
    video_file.close()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:video/mp4;base64,{data_url}");
            background-size: cover;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Main Streamlit app
st.title("My App with Background Video")  # Your app title

# Add the background video (replace 'your_video.mp4' with your actual video file)
add_bg_video("video.mp4") 

# Rest of your Streamlit content
st.header("Some Content Over the Video")
st.write("This is some text that will appear on top of the background video.")

# Add more content, widgets, charts, etc. as needed.
