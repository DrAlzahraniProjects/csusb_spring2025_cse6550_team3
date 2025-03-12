# Use a lightweight Python image
FROM python:3.10-slim-bookworm

# Install dependencies for running Apache, Jupyter, and Streamlit
RUN apt-get update && \
    apt-get install -y \
    apache2 \
    apache2-utils \
    && apt-get clean

# Install the required Apache modules for proxy and WebSocket support
RUN apt-get update && \
    apt-get install -y \
    libapache2-mod-proxy-uwsgi \
    libxml2-dev \
    libxslt-dev \
    && apt-get clean

# Set working directory
WORKDIR /app

# Install dependencies for building certain Python packages (like langchain with GROQ support)
RUN apt-get update && apt-get install -y gcc

# Copy only requirements first to leverage Docker caching
COPY requirements.txt /app/

# Install dependencies with no cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python code into the Docker container
COPY Chatbot.ipynb /app
COPY app.py /app

# Copy the rest of the app files
COPY . /app/

# Expose necessary ports for Streamlit and Jupyter
EXPOSE 2503
EXPOSE 2513

# Set up the Apache proxy configurations
RUN echo "ProxyPass /team3s25/jupyter http://localhost:2513/team3s25/jupyter" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPassReverse /team3s25/jupyter http://localhost:2513/team3s25/jupyter" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPass /team3s25 http://localhost:2503/team3s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPassReverse /team3s25 http://localhost:2503/team3s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "RewriteRule /team3s25/(.*) ws://localhost:2503/team3s25/$1 [P,L]" >> /etc/apache2/sites-available/000-default.conf

# Enable Apache modules for proxy and WebSocket support
RUN a2enmod proxy proxy_http rewrite

# Start Apache, Streamlit, and Jupyter Notebook using `sh` in the CMD
CMD ["sh", "-c", "apache2ctl start & python3 -u go-paper-spider.py & streamlit run app.py --server.port=2503 --server.baseUrlPath=/team3s25 & jupyter notebook --port=2513 --ip=0.0.0.0 --NotebookApp.base_url=/team3s25/jupyter --NotebookApp.notebook_dir=/app --NotebookApp.token='' --allow-root"]
