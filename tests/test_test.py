import os

import pytest

from config import load_config_from_yaml
from prototype.pipelines.test import run_testing_pipeline


@pytest.fixture(scope="module")
def temp_dir(tmpdir_factory):
    temp_dir = tmpdir_factory.mktemp("temp_dir")
    yield temp_dir
    temp_dir.remove()


@pytest.mark.integration
def test_run_testing_pipeline(temp_dir):
    root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config = load_config_from_yaml(
        file_path=os.path.join(root_directory, "config.yaml"), cli_args={}
    )

    inference_results = run_testing_pipeline(
        config=config,
        directory=root_directory,
        test_data_file_name=config.test.test_data_file_name,
        test_data=None,
    )

    inference_results.to_csv(os.path.join(temp_dir, "predictions_test.csv"))

    assert os.path.isfile(
        os.path.join(root_directory, "predictions_test.csv")
    ), "Output file does not exist"
    assert len(inference_results) == 953, "Inference results have incorrect length"
    assert (
        inference_results.index.name == "productIdentifier"
    ), "Inference results have incorrect index"
