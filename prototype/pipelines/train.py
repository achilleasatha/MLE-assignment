import logging
import os
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from config import AppConfigSettings
from prototype.data_schemata import TRAINING_DATA_SCHEMA
from prototype.pipelines.pipeline_factory import (
    CLASSIFIER_STEP_NAME,
    VECTORIZER_STEP_NAME,
    PipelineFactory,
)
from prototype.pipelines.preprocessing import get_inputs
from prototype.utils import export_model, fetch_data


def run_training_pipeline(
    *,
    config: AppConfigSettings,
    directory: str | os.PathLike,
    training_data_file_name: str | os.PathLike,
    target_variable: str,
    export_model_filename: str | os.PathLike,
    seed: Optional[int] = 42,
    test_size: Optional[float] = 0.1,
    custom_stopwords: Optional[frozenset[str]] = None,
) -> None:

    # Fetch training data
    df_train = fetch_data(
        dir_name=directory,
        file_name=training_data_file_name,
        schema=TRAINING_DATA_SCHEMA,
    )
    ds_train, ds_validate = train_test_split(
        df_train,
        test_size=test_size,
        random_state=seed,
        stratify=df_train[target_variable].values,
    )

    # These steps are slow, it would be preferable to enforce data versioning and simply log the tags for the datasets
    # mlflow.log_input(mlflow.data.from_pandas(ds_train), context='train')
    # mlflow.log_input(mlflow.data.from_pandas(ds_validate), context='validation')

    x_train = get_inputs(ds_train)
    y_train = ds_train[target_variable].values
    x_validate = get_inputs(ds_validate)
    y_validate = ds_validate[target_variable].values

    # Setup and fit_predict
    classifier = PipelineFactory().build_classifier_pipeline(
        config=config, custom_stopwords=custom_stopwords
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_train)
    # Our classifier does not implement a fit_predict method but we could create a wrapper to enable this
    # y_pred = classifier.fit_predict(x_train, y_train)

    calculate_model_metrics(
        classifier=classifier,
        full_ground_truth=df_train["pattern"],
        y_true=y_train,
        y_pred=y_pred,
    )

    # Hyper-param tuning
    optimal_classifier = find_optimal_classifier(
        config=config, classifier=classifier, input_data=x_train, ground_truth=y_train
    )

    # Validation
    calculate_cross_validation_metrics(
        config=config,
        optimal_classifier=optimal_classifier,
        x=x_validate,
        y=y_validate,
    )

    # Export artifacts
    save_path_or_uri = export_model(
        model=optimal_classifier,
        export_dir=config.export.export_dir,
        filename=export_model_filename,
        export_format=config.export.export_format,
        config=config,
    )

    logging.log(logging.INFO, f"Model pushed to: {save_path_or_uri}")


def find_optimal_classifier(
    config: AppConfigSettings,
    classifier: Pipeline,
    input_data: pd.Series,
    ground_truth: pd.Series,
) -> Pipeline:

    startified_k_fold = StratifiedKFold(n_splits=config.train.n_splits_k_fold_search)

    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=config.train.grid_search_parameters,
        n_jobs=-1,
        cv=startified_k_fold,
        verbose=1,
    )
    grid_search.fit(input_data, ground_truth)

    push_optimal_model_parameter_metrics(config=config, grid_search=grid_search)
    return grid_search.best_estimator_


def calculate_model_metrics(
    classifier: Pipeline,
    full_ground_truth: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> None:
    accuracy = np.mean(y_pred == y_true)
    classification_report_results = classification_report(
        y_true, y_pred, output_dict=True
    )
    confusion_matrix_results = confusion_matrix(y_true, y_pred)

    ctab = pd.crosstab(index=full_ground_truth, columns="count")
    classifier_coefficients = classifier.named_steps[CLASSIFIER_STEP_NAME].coef_
    vectorizer_feature_names = np.array(
        classifier.named_steps[VECTORIZER_STEP_NAME].get_feature_names_out()
    )

    def calculate_model_sparcity(coefficients: np.ndarray) -> dict:
        sparsity_values = np.sum(coefficients == 0, axis=1) / coefficients.shape[1]
        return {index: sparsity for index, sparsity in zip(ctab.index, sparsity_values)}

    model_sparcity = calculate_model_sparcity(coefficients=classifier_coefficients)

    def calculate_coefficient_ranking(
        coefficients: np.ndarray, feature_names: np.array, top: int = 5
    ) -> dict:
        return {
            ctab.index[i]: feature_names[np.argsort(-coefficients[i])[:top]]
            for i in range(coefficients.shape[0])
        }

    highest_ranked_coefficients = calculate_coefficient_ranking(
        coefficients=classifier_coefficients,
        feature_names=vectorizer_feature_names,
        top=5,
    )

    push_model_training_metrics(
        accuracy=accuracy,
        classification_report_results=classification_report_results,
        confusion_matrix_results=confusion_matrix_results,
        model_sparcity=model_sparcity,
        highest_ranked_coefficients=highest_ranked_coefficients,
    )


def calculate_cross_validation_metrics(
    config: AppConfigSettings, optimal_classifier: Pipeline, x: pd.Series, y: pd.Series
) -> None:
    xval_score = cross_val_score(
        estimator=optimal_classifier,
        X=x,
        y=y,
        cv=config.train.n_splits_k_fold_validation,
        n_jobs=-1,
    )
    mean_xval_score = np.mean(xval_score)
    mlflow.log_metric("avg_cross_validation_score", float(mean_xval_score))


def push_model_training_metrics(
    accuracy: np.floating,
    classification_report_results: dict,
    confusion_matrix_results: np.ndarray,
    model_sparcity: dict,
    highest_ranked_coefficients: dict,
) -> None:
    mlflow.log_metric("accuracy", float(accuracy))

    for class_name, metrics in classification_report_results.items():
        if isinstance(metrics, dict):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(
                    f"classification_report_{class_name}_{metric_name}", metric_value
                )
        else:
            mlflow.log_metric(f"classification_report_{class_name}", metrics)

    mlflow.log_param("confusion_matrix", confusion_matrix_results)

    for key, value in model_sparcity.items():
        mlflow.log_metric(f"model_sparcity_{key}", value)

    mlflow.log_param("highest_ranked_coefficients", highest_ranked_coefficients)


def push_optimal_model_parameter_metrics(
    config: AppConfigSettings, grid_search: GridSearchCV
) -> None:
    best_score = grid_search.best_score_
    mlflow.log_metric("grid_search_best_score", best_score)

    best_parameters = grid_search.best_estimator_.get_params()

    grid_search_parameters = config.train.grid_search_parameters or {}
    for param_name in sorted(grid_search_parameters.keys()):
        param_value = best_parameters[param_name]
        mlflow.log_param(f"optimal_{param_name}", param_value)


def convert_lists_to_tuples(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list):
            output_dict[key] = tuple(
                convert_lists_to_tuples(v) if isinstance(v, list) else v for v in value
            )
        else:
            output_dict[key] = value
    return output_dict
