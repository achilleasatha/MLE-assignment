import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import boto3
import joblib
import mlflow
import onnx
import onnxruntime
import pandas as pd
import pandera as pa
from skl2onnx import convert_sklearn
from sklearn.base import BaseEstimator

from config import AppConfigSettings


def fetch_data(
    dir_name: str | os.PathLike,
    file_name: str | os.PathLike,
    schema: Optional[pa.DataFrameSchema] = None,
) -> pd.DataFrame:
    """Fetches data from a specified file and directory, optionally validates
    it against a given schema.

    Args:
        dir_name (str or os.PathLike): Directory where the data file is located.
        file_name (str or os.PathLike): Name of the data file.
        schema (Optional[pa.DataFrameSchema], optional): Schema to validate the data against. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the fetched data.

    Raises:
        pa.errors.SchemaError: If the data does not match the specified schema.
        FileNotFoundError: If the specified file or directory does not exist.
    """
    try:
        df = pd.read_table(f"{dir_name}/{file_name}")
        if schema:
            schema.validate(df)
        else:
            logging.warning("Using unvalidated dataframe.")
        return df
    except (pa.errors.SchemaError, FileNotFoundError) as e:
        logging.error(e)
        sys.exit(1)


def export_model(
    model: BaseEstimator,
    export_dir: Optional[str | os.PathLike],
    filename: str | os.PathLike,
    config: AppConfigSettings,
    export_format: str = "pkl",
) -> str:
    """Export a scikit-learn model to the specified format.

    Args:
        model (BaseEstimator): The scikit-learn model to export.
        export_dir (str | os.PathLike): The directory to export the model to
        filename (str): The filename to save the exported model to.
        config (AppConfigSettings): our app config
        export_format (str): The format to export the model to. Supported formats: 'pkl' (default) or 'onnx'.
        push_to_service (str | None): if specified will upload the model to the relevant service
    Raises:
        ValueError: If an unsupported export format is specified.
    Returns:
        str: The exported model URI or local dir path
    """
    if not str(filename).endswith(export_format):
        filename = str(filename) + f".{export_format}"
    export_path = os.path.join(export_dir or "", filename)
    match export_format:
        case "pkl":
            joblib.dump(model, export_path)
        case "onnx":
            onnx_model = convert_sklearn(model)
            onnx.save_model(onnx_model, export_path)
        case _:
            raise ValueError(f"Unsupported export format: {export_format}")

    if (
        config.export.export_location is not None
        and config.export.export_location != "local"
    ):
        export_path = push_model_to_service(
            filename=filename, service=config.export.export_location, config=config
        )
        os.remove(export_path)
    return export_path


def push_model_to_service(
    filename: str | os.PathLike, service: str, config: AppConfigSettings
) -> str:
    """Push the exported model file to the specified service.

    Args:
        filename (str): The filename of the exported model.
        service (str): The name of the service to push the model to ('s3' or 'mlflow').
        service_config (dict): Configuration parameters for the service.

    Returns:
        :param filename: name of the file to export
        :param service: the service we'd like to use to export
        :param config: our app config
    """
    match service:
        case "s3":
            s3_bucket = config.export.s3_bucket or ""
            s3_key = config.export.s3_prefix or "" + Path(filename).name

            s3_client = boto3.client("s3")
            s3_client.upload_file(filename, s3_bucket, s3_key)

            model_uri = f"s3://{s3_bucket}/{s3_key}"

        case "mlflow":
            run_id = mlflow.active_run().info.run_id
            experiment_id = mlflow.active_run().info.experiment_id
            mlflow.log_artifact(f"{filename}.{config.export.export_format}")
            model_uri = (
                f"{config.mlflow.host}:{config.mlflow.port}/experiments/{experiment_id}"
                f"/runs/{run_id}/artifacts/{Path(filename).name}"
            )

        case _:
            raise ValueError(f"Unsupported service: {service}")

    return model_uri


def load_classifier(
    config: AppConfigSettings,
    directory: Optional[str | os.PathLike],
) -> Any:
    """Load a pre-trained sklearn model from the specified location and format.

    Args:
        config (AppConfigSettings): Configuration settings.
        location (str): The location from which to load the model ('local', 's3', or 'mlflow').
        directory (Optional[str | os.PathLike]): Optional directory path.

    Returns:
        Any: The loaded pre-trained model.
    """
    temp = False
    location = config.inference.classifier_location
    if location == "local":
        model_path = os.path.join(
            str(directory), str(config.inference.classifier_file_path or "")
        )
    elif location == "s3":
        s3_bucket = config.export.s3_bucket
        s3_key = (
            config.export.s3_prefix
            or "" + Path(str(config.inference.classifier_file_path or "")).name
        )
        s3_client = boto3.client("s3")
        temp_dir = tempfile.TemporaryDirectory()
        model_path = os.path.join(
            temp_dir.name, Path(str(config.inference.classifier_file_path)).name
        )
        with temp_dir as _:
            s3_client.download_file(s3_bucket, s3_key, model_path)
    elif location == "mlflow":
        artifact_path = (
            f"runs:/{config.inference.classifier_run_id}"
            f"/{config.inference.classifier_experiment_id}.{config.inference.format}"
        )
        output_path = "./local_model.pkl"
        mlflow.artifacts.download_artifacts(
            artifact_path=artifact_path, dst_path=output_path
        )
        temp = True
        model_path = os.path.join(str(directory), output_path)
    else:
        raise ValueError(f"Model loading location not recognized: {location}")
    if config.inference.format == "pkl":
        model = joblib.load(model_path)
    elif config.inference.format == "onnx":
        model = onnxruntime.InferenceSession(model_path)
    else:
        raise ValueError(f"Model format not recognized: {config.inference.format}")
    if temp:
        os.remove(model_path)
    logging.info(f"Loaded model from {model_path}")
    return model
