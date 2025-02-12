import os
from flask import Flask, request, jsonify
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Initialize Flask application
app = Flask(__name__)

# Set your OpenAI API key (or use environment variable for security)
os.environ["OPENAI_API_KEY"] = "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"  # Set your OpenAI API key

# Initialize the LLM model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # or gpt-4

# Initialize chat history
messages = [SystemMessage(content="You are a helpful assistant.")]

# Define the route for the root URL ("/")
@app.route("/", methods=["GET"])
def main():
    return 'Hello World!\n--Team 3'  # Returning a simple response

# Define the route for chat interaction (POST)
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")  # Get the message from the request JSON
    
    if user_input:
        # Add user's input to the chat history
        messages.append(HumanMessage(content=user_input))

        # Get the response from the LLM
        response = llm(messages)
        
        # Add the bot's response to the chat history
        messages.append(AIMessage(content=response.content))

        # Return the bot's response as a JSON response
        return jsonify({"response": response.content})
    else:
        return jsonify({"error": "No message provided"}), 400

# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 2503))  # Set port from environment variable or default to 2503
    app.run(debug=True, host='0.0.0.0', port=port)
