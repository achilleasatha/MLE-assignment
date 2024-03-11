import logging
import os
import signal
import subprocess  # nosec
import uuid

import mlflow

from config import AppConfigSettings


def mlflow_setup(config: AppConfigSettings, experiment_name: str | None, **kwargs):
    pid = start_process(config)
    log_run_config(config, experiment_name)
    log_run_tags(**kwargs)
    return pid


def start_process(config: AppConfigSettings):
    host = config.mlflow.host
    port = config.mlflow.port
    # Check if MLflow UI is already running
    if subprocess.run(
        ["/usr/bin/lsof", "-ti", f":{port}"], capture_output=True  # nosec
    ).stdout.strip():
        logging.info("MLflow UI is already running.")
        pid = None
    else:
        # Start MLflow UI server in the background
        process = subprocess.Popen(
            ["mlflow", "ui", "--host", host, "--port", str(port)],  # nosec
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        pid = process.pid
        logging.info(
            f"MLflow UI server started in the background at http://{host}:{port}"
        )
    mlflow.set_tracking_uri("http")
    return pid


def log_run_config(config: AppConfigSettings, experiment_name: str | None) -> None:
    if not experiment_name:
        experiment_name = f"{str(uuid.uuid4())}"
    mlflow.set_experiment(experiment_name)
    mlflow.log_params(config.dict())


def log_run_tags(**kwargs) -> None:
    # If we have a specific tagging strategy we can add relevant tags here
    tags = kwargs.pop("tags", {})
    for key, val in tags.items():
        mlflow.set_tag(key, val)


def mlflow_teardown(pid: int | None) -> None:
    # if pid is None:
    #     for proc in psutil.process_iter(['pid', 'cmdline']):
    #         if 'mlflow' in proc.cmdline():
    #             pid = proc.pid
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
            logging.info(f"MLflow UI server with PID {pid} has been terminated.")
        except ProcessLookupError:
            logging.info(f"MLflow process with PID {pid} not found.")
    else:
        logging.warning("Could not find a relevant mlflow process to terminate")
