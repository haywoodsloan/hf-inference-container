FROM python:3-slim

EXPOSE 80
WORKDIR /api

COPY requirements.txt ./
RUN pip install -r ./requirements.txt

COPY main.py ./

ENV HF_HOME=/api/.cache
ENV UVICORN_HOST="0.0.0.0"
ENV UVICORN_PORT="80"

ENTRYPOINT ["uvicorn", "main:app"]