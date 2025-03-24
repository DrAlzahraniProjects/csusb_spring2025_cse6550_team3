FROM python:3.10-slim-bookworm
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2 \
    libxslt1.1 \
    gcc \
    g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py go-paper-spider.py .

# Expose port
EXPOSE 2503

# Run both scraper and Streamlit
CMD ["sh", "-c", "python3 -u go-paper-spider.py & streamlit run app.py --server.port=2503 --server.baseUrlPath=/team3s25"]
