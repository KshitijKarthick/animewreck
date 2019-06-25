FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

ARG DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /animewreck/

RUN apt-get update && apt-get install -y \
    libpython3-dev \
    python-pip

RUN pip install --upgrade pip && \
    pip install -r /animewreck/requirements.txt

EXPOSE 80
WORKDIR /animewreck
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "9000"]
