# Stage 1: Build Stage
FROM python:3.10-slim-bookworm AS builder
WORKDIR /app
# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# Install torch CPU-only first, then the rest
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && find /usr/local/lib/python3.10/site-packages -name '*.so' -exec strip --strip-unneeded {} \; \
    && find /usr/local/lib/python3.10/site-packages -type d -name "tests" -exec rm -rf {} + \
    && find /usr/local/lib/python3.10/site-packages -type d -name "__pycache__" -exec rm -rf {} + \
    && find /usr/local/lib/python3.10/site-packages -type f -name "*.pyc" -exec rm -f {} +

# Stage 2: Final Stage
FROM python:3.10-slim-bookworm
WORKDIR /app
# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libxml2 \
    libxslt1.1 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/streamlit /usr/local/bin/streamlit
COPY --from=builder /usr/local/bin/scrapy /usr/local/bin/scrapy
COPY app.py /app/
EXPOSE 2503
CMD ["streamlit", "run", "app.py", "--server.port=2503", "--server.baseUrlPath=/team3s25"]
