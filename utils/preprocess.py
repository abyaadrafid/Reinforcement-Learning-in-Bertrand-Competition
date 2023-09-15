import itertools

import pandas as pd
from finrl.config import (
    INDICATORS,
    TRADE_END_DATE,
    TRADE_START_DATE,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
)
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split


def process_dataset():
    """
    Reads yahoo trading dataset from local store, preprocesses, splits
    and returns as train and validation sets
    """

    df = pd.read_pickle("datasets/yahoo1023.pkl")
    df.sort_values(["date", "tic"], ignore_index=True).head()
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])

    processed_full = processed_full.fillna(method="bfill")
    processed_full.sort_values(["date", "tic"], ignore_index=True).head(10)

    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    valid = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    return train, valid
