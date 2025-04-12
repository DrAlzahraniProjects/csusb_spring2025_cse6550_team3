#!/bin/bash

# Start Apache in the background
apache2ctl start

# Start your Scrapy spider in the background
python3 -u go-paper-spider.py &

# Start Streamlit in the background
streamlit run app.py --server.port=2503 --server.baseUrlPath=/team3s25 &

# This line keeps the container alive and makes docker ps show "python app.py"
python app.py

