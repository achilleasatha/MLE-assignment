![GitHub Actions](https://github.com/asos-talent/candidate-REF3103AAF/actions/workflows/ci.yaml/badge.svg)
![GitHub Actions](https://github.com/asos-talent/candidate-REF3103AAF/actions/workflows/build.yaml/badge.svg)

# ASOS MLE Takehome Assignment

### Using the CLI & Launching Jobs
 To run locally follow the setup instructions and simply run:
```bash
poetry run run_cli -j <job_name> [--optional-args args]
```
`<job_name>` can be either of `training, test, inference` which will
launch the relevant task.

For a list of optional args and their use refer to [prototype/cli/cli.py'](prototype/cli/cli.py)
- For training it will attempt to train, hyper-parameter tune with grid-search
a classifier model and export the model artifact along with logging metrics
- For test it will load a pre-trained model and run inference on the default or
specified data file
- For inference it will start a local server that will serve requests

### Example workflow

Running steps 1 and 3 is the quickest way to get a local server up and running
that will serve the model trained at step 1 and be able to serve requests.

```bash
# This will train our classifier and export it at project root with the specified name
poetry run run_cli -j training --export-model-filename trained_model.pkl

# This will load our pre-trained model and run inference on our test data (lands predictions_test.csv at root unless
# configured otherwise)
poetry run run_cli -j test --classifier-file-path trained_model.pkl

# This will load our pre-trained model and run a FastAPI app to serve the model
poetry run run_cli -j inference --classifier-file-path trained_model.pkl
```

Once the app is up and running you can go to [Docs](http://localhost:8000/docs)
for reference on how you can interact with the API and post inference requests

For example:
```bash
# For a single product
curl -X 'POST' \
  'http://localhost:8000/infer' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": [
    {
      "name": "string",
      "description": "Some text description",
      "product_id": 0
    }
  ]
}'

# For multiple products
curl -X 'POST' \
  'http://localhost:8000/infer' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": [
      {
      "name": "string",
      "description": "Some text description",
      "product_id": 0
    },
    {
      "name": "string",
      "description": "Some text description 2",
      "product_id": 1
    }
  ]
}'
```

### Running virtualized

All of the above can also be executed containerized without the need to manage
local envs:
```bash
docker build -t <image_name> .

docker run -p 8000:8000 -p 5000:5000 --network host <image_name> --job-name <job_name> [--optional-args args]
```

This assumes that the `trained_model.pkl` artifact exists at project root at build time of the image.
This is provided for convenience but by no means is a tenable solution.

Preferably an external storage service for model artifacts, eg. a standalone MlFlow server or an
S3 bucket should be used and pointed to in the config.

You can work around this with mounting volumes in Docker and sourcing a new model from there
but I didn't get around to address this and documenting it, as it's quite an edge case.
```bash
# This will train the model and export it according to our configuration
docker run <image_name> --job-name training [--optional-args args]

# This will start serving our model specified in the config
docker run -p 8000:8000 -p 5000:5000 --network host <image_name> --job-name inference [--optional-args args]
```

### Monitoring, Alerting & Observability
Our app exposes the `metrics` endpoint which returns metrics in
Prometheus format. We could use a combination of Prometheus and Grafana
to create dashboards for live traffic, health status of the app etc.

Currently supported metrics:
- Request latency
- Request count
- Inference latency
- Error count
```bash
curl http://localhost:8000/metrics
```
These metrics are not currently being ingested anywhere. This would require a running
instance of Prometheus, which I did not cover here.

Similarly, to set up alerting we'd need to be ingesting these metrics and then
set up alerting rules in Prometheus. Which I did not cover here either.

### Logging


Optionally you can run a local ElasticSearch server for logging.
You'll have to set `elasticsearch.enabled: true` in
[config.yaml](config.yaml)

To set up the local server run the following script, which will start a
containerized version:
```bash
chmod +x ./prototype/observability/elastic_server.sh

./prototype/observability/elastic_server.sh
```
This will provide you with a token which in turn you need to run:
```bash
docker run -e "ENROLLMENT_TOKEN=<token>" docker.elastic.co/elasticsearch/elasticsearch:8.12.2
```

### Load testing
Once the service is up and running we can perform a quick load test with
```bash
locust -f locustfile.py --headless --users 10000 --spawn-rate 100 --run-time 1h
```
On some consumer grade hardware used for local testing (i5-13500) with 20 workers
we could comfortably serve 50 req/s. The limiting factor being maximum
open file limit imposed by our operating system. This is adjustable though.

We could also apply caching so we can serve identical requests without having to
rerun inference.

Additionally, we could limit max number of allowed concurrent connections to avoid risks.

And of course, think about any potential optimisations we can make to our app.

## Project Setup

To set up the project locally follow these steps.

### Prerequisites

Make sure you have the following tools installed on your system:

- [Python](https://www.python.org/) (version 3.10)
- [Poetry](https://python-poetry.org/) (Python dependency management tool)
- [Git](https://git-scm.com/) (Version control system)
- [pre-commit](https://pre-commit.com/) (Git hook management tool)

Ideally install Poetry and pre-commit through **pipx** as this allows for global use.

```bash
pipx install poetry
```

```bash
pipx install pre-commit
```

It's recommended that a virtualenv is used for local setup & development.

Environment setup:
```bash
# Install pyenv
curl https://pyenv.run | bash
# Install pyenv-virtualenv plugin
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
# Install target python version
pyenv virtualenv 3.10.13
# Create venv
pyenv virtualenv 3.10.13 <env_name>
# Activate venv
pyenv activate <env_name>
# Set as default env to use in project root
pyenv local <env_name>
```

### Installation

1. **Clone the Repository:**

   ```bash
   git clone git@github.com:asos-talent/candidate-REF3103AAF.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd candidate-REF3103AAF
    ```

3. **Install Dependencies with Poetry:**

    ```bash
    poetry install
    ```
This command will install the project dependencies specified in pyproject.toml.


## Development
You can now start developing your project. Here are some useful commands:

### Set up pre-commit Hooks:

```bash
pre-commit install
```
This command will set up pre-commit hooks defined in .pre-commit-config.yaml to run before each commit,
ensuring code quality and consistency.

**Run Tests:**

```bash
poetry run pytest
```
Optionally specify ```-n=X``` where X is the number of CPU threads to distribute
workload on and this will depend on your specific hardware.

**Run pre-commit Hooks (Manually):**
```bash
pre-commit run --all
```

**Lint and Format Code:**
```bash
poetry run black .
poetry run isort .
```

### CI/CD
This project relies on GitHubActions to automate builds & tests when new code is pushed in the repo.

Currently, the option to upload CodeCov reports is disabled as for a private repo this should point to
some internal space. Similarly, pushing images to public DockerHub is disabled as this should point to
an internal and private image registry.

For some of these steps to work the relevant secrets / access tokens need to be set up.
```bash
#For CodeCov reports upload:
secrets.CODECOV_TOKEN

# For DockerHub Image push to Registry:
secrets.DOCKER_USERNAME
secrets.DOCKER_PASSWORD

# For release tagging
secrets.RELEASE_TOKEN
```

### Release with bumpver
This project follows SemVer standards for release tagging & versioning.

Refer to the relevant documentation of [bumpver](https://github.com/mbarkhau/bumpver)
and [SemVer](https://semver.org/) respectively for more info.

The format follows the convention: ```MAJOR.MINOR.PATCH[PYTAGNUM]```

For example:

```0.0.1 -> 0.0.2``` denotes the release of a patch

```0.0.1 -> 0.1.0``` denotes a minor release

```0.0.1 -> 1.0.0``` denotes a major release

Tags can be used to specify for example alpha and beta releases:

For example: ```0.0.1 -> 0.0.1a0``` is an alpha tagged release

In order to trigger a new release for a patch for example, with the option to add
a specific tag you can run:
```bash
SKIP=no-commit-to-branch bumpver update --patch --tag <tag>
```

The ```--final``` tag can be used to remove the release tag, this should be used for official releases
```bash
SKIP=no-commit-to-branch bumpver update --patch  --final
```

### Contributing
If you'd like to contribute to this project, please follow the Contributing Guidelines.

### Support
If you encounter any issues or have questions about this project, please open an issue on GitHub.
