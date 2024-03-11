import os
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    app_name: str = Field(..., description="Name of the application")
    debug: bool = Field(False, description="Enable debug mode")
    log_level: str = Field("info", description="Logging level")
    environment: str = Field(default="qa", description="Target environment")
    host: str = Field(..., description="Host of the application")
    port: int = Field(..., description="Port of the application")


class TrainConfig(BaseModel):
    data_dir: str = Field(..., description="Directory where data is stored")
    train_data_file_name: str = Field(..., description="Training data file name")
    target_variable: str = Field(..., description="Column name of target variable")
    seed: Optional[int] = Field(
        ..., description="Random seed for train/validation split"
    )
    test_size: Optional[float] = Field(
        ..., description="% of data to be used for validation"
    )
    n_splits_k_fold_search: Optional[int] = Field(
        ..., description="K-fold splits for param search"
    )
    n_splits_k_fold_validation: Optional[int] = Field(
        ..., description="K-fold splits for validation"
    )
    grid_search_parameters: Optional[dict] = Field(
        ..., description="Parameters to use for grid search"
    )


class TestConfig(BaseModel):
    data_dir: str = Field(..., description="Directory where data is stored")
    test_data_file_name: str = Field(..., description="Testing data file name")
    index: str = Field(..., description="Column name of index")


class InferenceConfig(BaseModel):
    classifier_run_id: Optional[str] = Field(
        ..., description="Model run id to load from Mlflow to serve for inference"
    )
    classifier_experiment_id: Optional[str] = Field(
        ..., description="The experiment id to load from Mlflow to serve for inference"
    )
    classifier_file_path: Optional[str | os.PathLike] = Field(
        ..., description="Path to load model from - can be S3 or local"
    )
    classifier_location: str = Field(
        ...,
        description="Location to source pre-trained model from can be local, s3, "
        "mlflow",
    )
    format: Optional[str] = Field(
        ...,
        description="Format model needs to be loaded in currently supported: [pkl, onnx]",
    )


class ClassifierConfig(BaseModel):
    classifier_name: str = Field(..., description="Name of classifier to be used")
    classifier_loss: Optional[str] = Field(
        ..., description="Loss function for classifier"
    )
    classifier_penalty: Optional[str] = Field(
        ..., description="Regularization term for classifier"
    )
    classifier_alpha: Optional[float] = Field(
        ..., description="Scaling for regularization term"
    )


class VectorizerConfig(BaseModel):
    vectorizer_name: Optional[str] = Field(
        None, description="Name of vectorizer to be used"
    )


class ScalerConfig(BaseModel):
    scaler_name: Optional[str] = Field(
        None, description="Name of scaling function to be used"
    )


class ExportConfig(BaseModel):
    export_dir: Optional[str | os.PathLike] = Field(
        ..., description="Dir to export results and model artifacts to"
    )
    s3_bucket: Optional[str] = Field(
        ..., description="S3 bucket to use for exporting results and artifacts"
    )
    s3_prefix: Optional[str] = Field(
        ..., description="S3 subdirectory to use for exporting results and artifacts"
    )
    export_format: str = Field(
        ..., description="Format to save model as. Currently supported: [pkl, onnx]"
    )
    export_location: Optional[str] = Field(
        ...,
        description="Location to store model artifact None assumes local,"
        "possible values ['s3', 'mlflow']",
    )


class MlflowConfig(BaseModel):
    host: str = Field(..., description="Mlflow host to connect to")
    port: int = Field(..., description="Port to connect to")


class ElasticsearchConfig(BaseModel):
    host: str = Field(..., description="Elasticsearch host URI to connect to")
    enabled: bool = Field(..., description="Enable or disable logging to elasticsearch")


class AppConfigSettings(BaseModel):
    app: AppConfig = Field(..., description="Application settings")
    train: TrainConfig = Field(..., description="Training settings")
    test: TestConfig = Field(..., description="Testing settings")
    inference: InferenceConfig = Field(..., description="Inference settings")
    classifier: ClassifierConfig = Field(
        ..., description="Model specific configuration"
    )
    vectorizer: VectorizerConfig = Field(
        ..., description="Vectorizer specific configuration"
    )
    scaler: ScalerConfig = Field(..., description="Scaler specific configuration")
    export: ExportConfig = Field(..., description="Export config")
    mlflow: MlflowConfig = Field(..., description="Mlflow server settings")
    elasticsearch: ElasticsearchConfig = Field(
        ..., description="Elasticsearch settings"
    )

    class Config:
        env_prefix = "APP_"  # Prefix for environment variables
        case_sensitive = False  # Case sensitivity for environment variables


# Load configuration data from a YAML file
def load_config_from_yaml(
    file_path: str, cli_args: Optional[dict]
) -> AppConfigSettings:
    """Load configuration data from a YAML file and merge it with CLI
    arguments.

    Args:
        file_path (str): Path to the YAML configuration file.
        cli_args (dict): Dictionary containing CLI arguments.

    Returns:
        AppConfigSettings: Merged configuration settings.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """

    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)

    with open(file_path, "r", encoding="utf-8") as yaml_file:
        config_data = yaml.load(
            yaml_file.read(), Loader=yaml.FullLoader  # nosec
        )  # we can't use safe_load because of tags

    # Merge configuration data with CLI arguments
    if cli_args is not None:
        for key, value in cli_args.items():
            for section in config_data.values():
                if key in section and value is not None:
                    section[key] = value
    return AppConfigSettings(**config_data)
