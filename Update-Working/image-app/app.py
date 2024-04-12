import streamlit as st
from PIL import Image
import os

def main():
    st.title("Photo Comparison App")

    folder1_path = "C:\\Users\\Ifham.Khan\\Desktop\\Final Year Project\\downloads\\frames"
    folder2_path = "C:\\Users\\Ifham.Khan\\Desktop\\Final Year Project\\downloads\\frames"

    folder1 = st.sidebar.selectbox("Select Folder 1", os.listdir(folder1_path))
    folder2 = st.sidebar.selectbox("Select Folder 2", os.listdir(folder2_path))

    if st.sidebar.button("Compare"):
        compare_folders(folder1_path, folder2_path, folder1, folder2)

def compare_folders(folder1_path, folder2_path, folder1, folder2):
    # Get the list of files in each folder
    files1 = os.listdir(os.path.join(folder1_path, folder1))
    files2 = os.listdir(os.path.join(folder2_path, folder2))

    # Compare the lists
    if files1 == files2:
        st.write("Folders contain the same images.")
    else:
        st.write("Folders do not contain the same images.")

if __name__ == "__main__":
    main()
