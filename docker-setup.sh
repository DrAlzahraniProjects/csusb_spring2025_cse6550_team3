#!/bin/bash

# Define container name
CONTAINER_NAME="team3s25-app"
PORT_NUM=2503
J_PORTNUM=2513

# Prompt user for API key
echo "Enter your Groq-API key:"
read GROQ_API_KEY

# Build the Docker image
echo "Building Docker image..."
docker build -t $CONTAINER_NAME .

# Running Docker Image
docker run -d -p $PORT_NUM:$PORT_NUM -p $J_PORTNUM:$J_PORTNUM --name $CONTAINER_NAME -e GROQ_API_KEY="$GROQ_API_KEY" $CONTAINER_NAME

# Output where the apps are running
echo "Streamlit is available at: http://localhost:$PORT_NUM/team3s25"
echo "Jupyter Notebook is available at: http://localhost:$J_PORTNUM/team3s25/jupyter"