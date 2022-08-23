FROM tensorflow/tensorflow:2.9.1

ARG ENVIRONMENT=production
ARG POETRY_VERSION=1.1.13

RUN pip install poetry==$POETRY_VERSION

WORKDIR /code

COPY pyproject.toml poetry.lock ./

RUN POETRY_VIRTUALENVS_CREATE=false poetry install

COPY src/retrain_bert .
