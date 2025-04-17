#!/bin/bash

# Define container name, volume, and port
CONTAINER_NAME="team3s25-app"
VOLUME_NAME="scrapy_output"
PORT_NUM=2503
DATA_DIR=$(pwd)/data_files

# Prompt user for API key
echo "Enter your Groq-API key:"
read -r GROQ_API_KEY

# Check if the volume exists
if docker volume ls -q -f name=$VOLUME_NAME | grep -q $VOLUME_NAME; then
    echo "Volume '$VOLUME_NAME' already exists."
else
    echo "Creating volume '$VOLUME_NAME'..."
    docker volume create $VOLUME_NAME
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t $CONTAINER_NAME .

# Check if a container with the same name is already running
if [ $(docker ps -q -f name=$CONTAINER_NAME) ]; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run container
echo "Running container..."
docker run -d \
  -p $PORT_NUM:$PORT_NUM \
  --name $CONTAINER_NAME \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  -v "$DATA_DIR":/app/data \
  -v $VOLUME_NAME:/data \
  $CONTAINER_NAME

# Output URL
echo "Streamlit is available at: http://localhost:$PORT_NUM/team3s25"
