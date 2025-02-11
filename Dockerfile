# Use a base image with CUDA for GPU acceleration (or change for CPU use)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install necessary Python libraries
RUN pip3 install torch transformers accelerate sentencepiece streamlit

# Set working directory
WORKDIR /app

# Copy the Streamlit app
COPY app.py /app/app.py

# Expose Streamlit's default port
EXPOSE 8501

# Install required dependencies
RUN pip install torch transformers accelerate sentencepiece llama-cpp-python streamlit

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
