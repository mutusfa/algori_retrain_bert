FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG ENVIRONMENT=production

COPY ./jupyter_server_config.json /root/.jupyter/jupyter_server_config.json

WORKDIR /code

RUN pip install --upgrade jsonschema  # Why do I need to fix deps manually

COPY pyproject.toml ./

COPY src/retrain_bert ./src/retrain_bert

RUN pip install .