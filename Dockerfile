# Use a lightweight Python base image
FROM python:3.10-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies in one layer, minimize size
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Copy only requirements file first (leverages caching)
COPY requirements.txt /app/

# Install Python dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary app files
COPY app.py go-paper-spider.py /app/

# Expose the port
EXPOSE 2503

# Run Streamlit directly (no Apache)
CMD ["streamlit", "run", "app.py", "--server.port=2503", "--server.baseUrlPath=/team3s25"]
