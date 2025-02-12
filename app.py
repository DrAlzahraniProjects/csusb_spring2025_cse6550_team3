import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Set your OpenAI API key (or use environment variable for security)
os.environ["OPENAI_API_KEY"] = "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"  # Set your OpenAI API key

# Initialize the LLM model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # or gpt-4

# Initialize chat history
messages = [SystemMessage(content="You are a helpful assistant.")]

# Streamlit app layout
st.title("Chatbot with LangChain")
st.write("Type your message and the chatbot will respond!")

# Input box for the user to type their message
user_input = st.text_input("You: ", "")

# If user provides input
if user_input:
    # Add user's input to the chat history
    messages.append(HumanMessage(content=user_input))

    # Get the response from the LLM
    response = llm(messages)

    # Add the bot's response to the chat history
    messages.append(AIMessage(content=response.content))

    # Display the bot's response
    st.write(f"Bot: {response.content}")
else:
    st.write("Please type something to start the conversation.")

