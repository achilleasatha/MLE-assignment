FROM python:3.10-slim

RUN apt-get update && apt-get install -y lsof
RUN python3 -m pip install --user pipx
ENV PATH="/root/.local/bin:${PATH}"
ENV GIT_PYTHON_REFRESH quiet
RUN pipx install poetry

WORKDIR /app
COPY . /app

RUN poetry install

# FastAPI and Mlflow ports
EXPOSE 8000
EXPOSE 5000

LABEL stage=intermediate
ENTRYPOINT ["poetry", "run", "run_cli"]

# Default command-line arguments
#CMD ["-j", "training", "--classifier-file-path", "trained_model.pkl", \
