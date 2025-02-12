import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Hugging Face model and tokenizer for LLaMA model
model_name = "meta-llama/Llama-2-7b-chat-hf"  # LLaMA model (ensure you have access to this model on Hugging Face)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate responses from the model
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI for chatbot
st.title("LLaMA Chatbot")
st.write("Talk to the chatbot powered by LLaMA (meta-llama)! Type 'exit' to end the conversation.")

# Chat history list to keep track of conversation
if "history" not in st.session_state:
    st.session_state.history = []

# Show the chat history and prompt user for input
for message in st.session_state.history:
    st.write(message)

# Chat input from the user
user_input = st.text_input("You: ", "")

# Display response from the model
if user_input:
    st.session_state.history.append(f"You: {user_input}")
    if user_input.lower() == "exit":
        st.write("Goodbye!")
    else:
        response = generate_response(user_input)
        st.session_state.history.append(f"Bot: {response}")
