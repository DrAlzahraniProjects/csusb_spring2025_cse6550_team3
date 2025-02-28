# Use a lightweight Python image
FROM python:3.10-slim-bookworm

# Set working directory
WORKDIR /app

# Install dependencies for building certain Python packages (like langchain with GROQ support)
RUN apt-get update && apt-get install -y gcc

# Copy only requirements first to leverage Docker caching
COPY requirements.txt /app/

# Install dependencies with no cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app files
COPY . /app/

# Expose necessary ports for Streamlit and Jupyter
EXPOSE 2503
EXPOSE 2513

# Start both Streamlit and Jupyter Notebook
CMD ["bash", "-c", "python3 go-spider.py & streamlit run app.py --server.port=2503 --server.address=0.0.0.0 & jupyter notebook --ip=0.0.0.0 --port=2513 --no-browser --NotebookApp.token='' --allow-root"]