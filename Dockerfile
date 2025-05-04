# Stage 1: Build Stage
FROM python:3.10-slim-bookworm AS builder
WORKDIR /app
# Install build dependencies, including BLAS/LAPACK
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# Update pip, install torch CPU-only first, then the rest
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && find /usr/local/lib/python3.10/site-packages -type d -name "tests" -exec rm -rf {} + \
    && find /usr/local/lib/python3.10/site-packages -type d -name "__pycache__" -exec rm -rf {} + \
    && find /usr/local/lib/python3.10/site-packages -type f -name "*.pyc" -exec rm -f {} +

# Stage 2: Final Stage
FROM python:3.10-slim-bookworm
WORKDIR /app
# Install runtime dependencies, including BLAS/LAPACK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libapache2-mod-proxy-uwsgi \
    libgomp1 \
    libxml2 \
    libxslt1.1 \
    apache2 \
    apache2-utils \
    zlib1g \
    libblas3 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/streamlit /usr/local/bin/streamlit
COPY --from=builder /usr/local/bin/scrapy /usr/local/bin/scrapy
COPY app.py /app/
COPY go-paper-spider.py /app/
COPY mycrawler /app/mycrawler
EXPOSE 2503

# Set up Apache proxy configurations
RUN echo "ProxyPass /team3s25 http://localhost:2503/team3s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPassReverse /team3s25 http://localhost:2503/team3s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "RewriteRule /team3s25/(.*) ws://localhost:2503/team3s25/$1 [P,L]" >> /etc/apache2/sites-available/000-default.conf

# Enable necessary Apache modules
RUN a2enmod proxy proxy_http rewrite

CMD ["python", "app.py"]
