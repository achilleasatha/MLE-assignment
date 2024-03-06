# ASOS MLE Takehome Assignment

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

### Set up pre-commit Hooks:

```bash
pre-commit install
```
This command will set up pre-commit hooks defined in .pre-commit-config.yaml to run before each commit,
ensuring code quality and consistency.

### Development
You can now start developing your project. Here are some useful commands:

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
