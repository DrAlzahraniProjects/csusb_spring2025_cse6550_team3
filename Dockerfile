# Build stage
FROM python:3.10-slim-bookworm AS builder
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/*

# Runtime stage
FROM python:3.10-slim-bookworm
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY app.py go-paper-spider.py /app/
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/*
EXPOSE 2503
CMD ["streamlit", "run", "app.py", "--server.port=2503", "--server.baseUrlPath=/team3s25"]
