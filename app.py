import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer (adjust model name if needed)
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("💬 LLaMA Chatbot")
st.write("Enter a prompt and get a response from LLaMA!")

user_input = st.text_area("Your prompt:")

if st.button("Generate Response"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("### Response:")
        st.write(response)
    else:
        st.warning("Please enter a prompt!")

