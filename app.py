import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Llama model
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Adjust model as needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

st.title("Llama 2 Chatbot")

user_input = st.text_area("Enter your prompt:")

if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    st.subheader("Response:")
    st.write(response)
