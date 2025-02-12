FROM python:3.10.0-slim-buster
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 2503
EXPOSE 8888

CMD ["bash", "-c", "streamlit run app.py --server.port=2503 --server.address=0.0.0.0 & jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=token --allow-root"]
