import logging
import os
import sys
from typing import Optional

import pandas as pd
import pandera

from config import AppConfigSettings
from prototype.data_schemata import INFERENCE_DATA_SCHEMA, TEST_DATA_SCHEMA
from prototype.pipelines.preprocessing import get_inputs
from prototype.utils import fetch_data, load_classifier


def run_testing_pipeline(
    *,
    config: AppConfigSettings,
    directory: str | os.PathLike,
    test_data_file_name: Optional[str | os.PathLike],
    test_data: Optional[pd.DataFrame],
) -> pd.Series:

    # Fetch data
    if test_data is None:
        if test_data_file_name is None:
            raise ValueError(
                "Either 'test_data' or 'test_data_file_name' must be provided."
            )
        test_data = fetch_data(
            dir_name=os.path.join(directory, config.test.data_dir),
            file_name=test_data_file_name,
            schema=TEST_DATA_SCHEMA,
        )
    else:
        logging.warning(
            f"Both 'test_data' and 'test_data_file_name: {test_data_file_name}' are provided. "
            f"'test_data' will be used, and 'test_data_file_name' will be ignored."
        )

    # Preprocess data
    # TODO wrap this in custom preprocessor
    x_test = get_inputs(test_data)

    # Fetch pre-trained classifier
    classifier = load_classifier(config=config, directory=directory)

    model_format = config.inference.format
    if model_format == "pkl":
        inference_results = classifier.predict(x_test)
    elif model_format == "onnx":
        inference_results = classifier.run(None, {"input": x_test})
    else:
        raise ValueError(f"Model format not recognized: {config.inference.format}")
    try:
        inference_results = pd.Series(
            inference_results, index=test_data[config.test.index]
        )
        INFERENCE_DATA_SCHEMA.validate(inference_results)
        return inference_results
    except (pandera.errors.SchemaError, FileNotFoundError) as e:
        logging.error(e)
        sys.exit(1)
