# streamlit_pages.py
import streamlit as st

class MultiPage:
    def __init__(self):
        self.pages = {}

    def add_page(self, name, func):
        self.pages[name] = func

    def run(self):
        page = st.sidebar.selectbox('Navigation', list(self.pages.keys()))
        self.pages[page]()


