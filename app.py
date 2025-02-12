import os
import gdown
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Download the .gguf file from Google Drive
# Replace with your Google Drive link (after extracting the file ID)
file_url = 'https://drive.google.com/uc?id=YOUR_FILE_ID'  # Replace YOUR_FILE_ID with the actual file ID
output_path = 'my_model.gguf'

gdown.download(file_url, output_path, quiet=False)

# 2. Load the tokenizer and model from the local .gguf file
# Assuming you have a Hugging Face-compatible model architecture
# You can replace the 'AutoModelForCausalLM' with the model class that matches your .gguf file.

# Define the path to your downloaded model (this could be your local directory if you downloaded the model manually)
model_directory = './'  # Where the .gguf model file is saved

# Initialize model and tokenizer (customize as per your model architecture)
# If your model uses specific layers, you may need to adjust the tokenizer and model loading.

tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

# 3. Set up the chatbot
print("Chatbot ready! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="pt")

    # Generate a response from the model
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)

    # Decode the response
    bot_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Bot: {bot_output}")
