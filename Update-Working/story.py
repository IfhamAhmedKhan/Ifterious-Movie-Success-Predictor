import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer (same as before)
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Streamlit app
st.title("Text Generation with GPT-2")
st.write("Powered by Hugging Face Transformers")

# User input (same as before)
prompt = st.text_input("Enter a prompt:", "Last night, I made progress on my game development project by")

if st.button("Generate Text"):
    # Tokenize the input text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text with adjustments
    output = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,  # Still prevents word repetition, but less strictly
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2,   # Penalizes repeated sequences (not just words)
        num_beams=3,             # Explore multiple generation paths
        early_stopping=False    # Let the model complete the story
    )

    # Decode and display the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write(generated_text)
