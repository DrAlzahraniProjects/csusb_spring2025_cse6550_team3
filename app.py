from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os

# Set your OpenAI API key (make sure to replace it with your own API key)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"  # Replace with your OpenAI API key

# Initialize OpenAI model (no need to use any Hugging Face or other models)
llm = OpenAI(model_name="gpt-3.5-turbo")  # Or "gpt-4" for GPT-4

# Chat history initialization
messages = [SystemMessage(content="You are a helpful assistant.")]

print("Chatbot ready! Type 'exit' to quit.")

# Main loop for chat
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Append user message to chat history
    messages.append(HumanMessage(content=user_input))

    # Get the model's response based on chat history
    response = llm(messages)

    print("Bot:", response.content)

    # Append model's response to chat history for context
    messages.append(AIMessage(content=response.content))
