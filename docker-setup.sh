#!/bin/bash

# Define container name
CONTAINER_NAME="team3s25-app"

# Prompt user for API key
echo "Enter your Groq-API key:"
read GROQ_API_KEY

# Build the Docker image
echo "Building Docker image..."
docker build -t $CONTAINER_NAME .

# Check if a container with the same name is already running
if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run the Docker container
echo "Running Docker container..."
docker run -d -p 2503:2503 -p 2513:2513 --name $CONTAINER_NAME -e GROQ_API_KEY="$GROQ_API_KEY" $CONTAINER_NAME

echo "Docker container is running. Access your services at:"
echo "- Streamlit: http://localhost:2503"
echo "- Jupyter Notebook: http://localhost:2513"
