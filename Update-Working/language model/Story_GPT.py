# Import necessary libraries
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to generate story using GPT-2
def generate_story(idea):
    input_ids = tokenizer.encode(idea, return_tensors='pt')
    output = model.generate(input_ids, max_length=300, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

# Streamlit app layout
st.title("Story Generator")

# Input form for the story idea
idea = st.text_input("Enter your story idea:")

# Button to generate story
if st.button("Generate Story"):
    if idea:
        with st.spinner('Generating story...'):
            story = generate_story(idea)
        st.subheader("Generated Story:")
        st.write(story)
    else:
        st.error("Please enter a story idea.")
