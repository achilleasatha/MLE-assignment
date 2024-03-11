import re

import pandas as pd


def get_inputs(df: pd.DataFrame) -> pd.Series:
    x_train = df["name"].str.cat(df["description"], sep=";")  # .values
    return sanitize_input_features(input_data=x_train)


def sanitize_input_features(input_data: pd.Series) -> pd.Series:
    def remove_numbers_from_text(input_series: pd.Series) -> pd.Series:
        # for i in range(len(input_series)):
        #     input_series[i] = re.sub(r"[0-9]+", "", input_series[i])
        return input_series.apply(lambda x: re.sub(r"[0-9]+", "", x))

    return remove_numbers_from_text(input_data)
