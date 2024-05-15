import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def generate_text(prompt, max_length, temperature, num_beams):
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            repetition_penalty=1.2,
            num_beams=num_beams,
            early_stopping=True
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        return str(e)

# Streamlit app
st.title("Story Generation with GPT-2")

# User input
prompt = st.text_input("Enter a prompt:", "Last night, I made progress on my game development project by")

# Additional generation parameters
max_length = st.slider("Max Length", 50, 500, 200)
temperature = st.slider("Temperature", 0.5, 1.5, 0.8)
num_beams = st.slider("Number of Beams", 1, 5, 3)

if st.button("Generate Text"):
    with st.spinner("Generating..."):
        generated_text = generate_text(prompt, max_length, temperature, num_beams)
        st.write(generated_text)
