FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.3.2

COPY pyproject.toml poetry.lock /animewreck/

WORKDIR /animewreck

RUN apt-get update && apt-get install -y \
    libpython3-dev \
    python3-pip

RUN pip install --upgrade pip && \
    pip install poetry=="$POETRY_VERSION"

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

EXPOSE 9000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "9000"]
