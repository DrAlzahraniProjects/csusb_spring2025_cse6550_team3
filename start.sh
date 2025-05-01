#!/bin/bash

# Start Apache in the background
apache2ctl start &

# Start the spider in the background
python3 -u go-paper-spider.py &

# Start the Streamlit app in the foreground
exec streamlit run app.py --server.port=2503 --server.baseUrlPath=/team3s25

