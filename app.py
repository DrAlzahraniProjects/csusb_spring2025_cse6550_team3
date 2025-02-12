import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set your Hugging Face model name
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Use your model here if different

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move the model to CPU (since no GPU is available on your Mac)
model = model.to("cpu")  # Force the model to use the CPU

# Function to get chatbot response
def get_chatbot_response(prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate a response from the model
    outputs = model.generate(**inputs)
    
    # Decode the generated output to a string
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit interface to interact with the chatbot
import streamlit as st

# Set the page title for the Streamlit app
st.title("LLaMA Chatbot")

# Add a text box for user input
user_input = st.text_input("You:", "")

# Generate and display response
if user_input:
    chatbot_response = get_chatbot_response(user_input)
    st.write(f"Bot: {chatbot_response}")

# Running the Streamlit app
if __name__ == "__main__":
    st.write("Chatbot is ready! Type your message.")
