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

**Run pre-commit Hooks (Manually):**
```bash
poetry run pre-commit run --all-files
```

**Lint and Format Code:**
```bash
poetry run black .
poetry run isort .
```

### Contributing
If you'd like to contribute to this project, please follow the Contributing Guidelines.

### Support
If you encounter any issues or have questions about this project, please open an issue on GitHub.
