import json
import os
import subprocess  # nosec
import sys

import click
import mlflow

from prototype.logger import setup_logging
from prototype.observability.mlflow import mlflow_setup, mlflow_teardown
from prototype.pipelines.test import run_testing_pipeline
from prototype.pipelines.train import run_training_pipeline

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from config import load_config_from_yaml  # noqa

POSSIBLE_JOBS = ["training", "test", "inference"]


@click.command()
@click.option("--config-file", default="config.yaml", help="Path to configuration file")
@click.option("--app-name", default=None, help="Name of the application")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
@click.option("--log-level", default="info", help="Logging level")
@click.option("--experiment-name", default=None, help="Name of the experiment")
@click.option(
    "--output-dir",
    "-o",
    default="",
    type=click.Path(dir_okay=True, file_okay=False),
    help="Output dir",
)
@click.option(
    "--job-name",
    "-j",
    type=click.Choice(POSSIBLE_JOBS),
    default="training",
    help="Name of the job to run",
)
@click.option("--target-variable", help="Target variable used for training")
@click.option(
    "--export-model-filename",
    type=click.Path(dir_okay=False, file_okay=True),
    help="Export model filename",
)
@click.option(
    "--test-data-file-name",
    default=None,
    type=click.Path(dir_okay=False, file_okay=True),
    help="Test data",
)
@click.option("--classifier-file-path", default=None, type=click.Path(dir_okay=False))
def run_cli(
    config_file,
    app_name,
    debug,
    log_level,
    experiment_name,
    output_dir,
    job_name,
    target_variable,
    export_model_filename,
    test_data_file_name,
    classifier_file_path,
):
    """Command-line interface for the application."""
    cli_args = {k: v for k, v in locals().items() if v is not None}
    config = load_config_from_yaml(config_file, cli_args)
    # click.secho(json.dumps(config.dict(), indent=2), fg="yellow")

    logger = setup_logging(config, job_name)
    logger.info(f"Config:\n{json.dumps(config.dict(), indent=2)}")

    # We want to set the project root at the root of the repo
    root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if job_name == "training":
        mlflow_pid = mlflow_setup(config, experiment_name)
        run_training_pipeline(
            config=config,
            directory=os.path.join(root_directory, config.train.data_dir),
            training_data_file_name=config.train.train_data_file_name,
            target_variable=config.train.target_variable,
            export_model_filename=export_model_filename
            or experiment_name
            or mlflow.active_run().info.experiment_id,
        )
        mlflow_teardown(pid=mlflow_pid)
    elif job_name == "test":
        inference_results = run_testing_pipeline(
            config=config,
            directory=root_directory,
            test_data_file_name=test_data_file_name or config.test.test_data_file_name,
            test_data=None,
        )
        inference_results.to_csv(
            os.path.join(root_directory, output_dir, "predictions_test.csv")
        )
        logger.info(
            f'Results saved to {os.path.join(root_directory, output_dir, "predictions_test.csv")}'
        )
    elif job_name == "inference":
        subprocess.run(  # nosec
            f"uvicorn main:app --reload --host {config.app.host} --port {config.app.port}",
            shell=True,
            cwd=os.path.join(root_directory, "prototype"),
        )
    if debug:
        breakpoint()


if __name__ == "__main__":
    run_cli()
