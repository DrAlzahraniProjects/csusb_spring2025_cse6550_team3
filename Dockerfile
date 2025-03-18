# Use a lightweight Python base image
FROM python:3.10-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies required for the chatbot
RUN apt-get update && apt-get install -y \
    gcc \
    libapache2-mod-proxy-uwsgi \
    libxml2-dev \
    libxslt-dev \
    apache2 \
    apache2-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only requirements file first (leverages Docker caching)
COPY requirements.txt /app/

# Install Python dependencies without cache to reduce size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/

# Expose the necessary ports
EXPOSE 2503

# Set up Apache proxy configurations
RUN echo "ProxyPass /team3s25 http://localhost:2503/team3s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPassReverse /team3s25 http://localhost:2503/team3s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "RewriteRule /team3s25/(.*) ws://localhost:2503/team3s25/$1 [P,L]" >> /etc/apache2/sites-available/000-default.conf

# Enable necessary Apache modules
RUN a2enmod proxy proxy_http rewrite

# Start Apache and Streamlit
CMD ["sh", "-c", "apache2ctl start & streamlit run app.py --server.port=2503 --server.baseUrlPath=/team3s25"]
