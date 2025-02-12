from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os

# Set your API key (use environment variables in production)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Initialize the LLM model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # or gpt-4

# Chat history
messages = [SystemMessage(content="You are a helpful assistant.")]

print("Chatbot ready! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    messages.append(HumanMessage(content=user_input))

    response = llm(messages)
    print("Bot:", response.content)

    messages.append(AIMessage(content=response.content))

