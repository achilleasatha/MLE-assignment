import os
import shutil
import tempfile

from config import load_config_from_yaml
from prototype.pipelines.train import run_training_pipeline


def test_training_pipeline_integration():
    root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config = load_config_from_yaml(
        file_path=os.path.join(root_directory, "config.yaml"), cli_args={}
    )
    training_data_file_name = config.train.train_data_file_name
    export_model_filename = "test_model.pkl"
    temp_dir = tempfile.mkdtemp()
    config.export.export_dir = temp_dir

    try:
        run_training_pipeline(
            config=config,
            directory=os.path.join(root_directory, config.train.data_dir),
            training_data_file_name=training_data_file_name,
            target_variable=config.train.target_variable,
            export_model_filename=export_model_filename,
            seed=42,
            test_size=0.1,
            custom_stopwords=None,
        )

        # Check if the model artifact is exported to the specified directory with the expected file structure
        expected_model_path = os.path.join(temp_dir, export_model_filename)
        assert os.path.exists(
            expected_model_path
        ), f"Model artifact {export_model_filename} not found in the export directory"
    finally:
        shutil.rmtree(temp_dir)
