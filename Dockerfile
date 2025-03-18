# Stage 1: Build stage
FROM python:3.10-slim-bookworm AS build

WORKDIR /app

# Install only build dependencies (gcc, libxml2-dev, libxslt-dev for building Python dependencies)
RUN apt-get update && apt-get install -y gcc libxml2-dev libxslt-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to leverage Docker cache for pip install
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

# Stage 2: Final image
FROM python:3.10-slim-bookworm

WORKDIR /app

# Copy installed Python packages and application files from the build stage
COPY --from=build /app /app

# Install Apache only in the final image, if necessary
RUN apt-get update && apt-get install -y apache2 apache2-utils libapache2-mod-proxy-uwsgi && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up Apache proxy configurations (if needed)
RUN echo "ProxyPass /team3s25 http://localhost:2503/team3s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPassReverse /team3s25 http://localhost:2503/team3s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "RewriteRule /team3s25/(.*) ws://localhost:2503/team3s25/$1 [P,L]" >> /etc/apache2/sites-available/000-default.conf

# Enable necessary Apache modules
RUN a2enmod proxy proxy_http rewrite

# Copy the rest of the application files
COPY . /app/

# Expose necessary ports
EXPOSE 2503

# Start Apache and Streamlit
CMD ["sh", "-c", "apache2ctl start & streamlit run app.py --server.port=2503 --server.baseUrlPath=/team3s25"]
