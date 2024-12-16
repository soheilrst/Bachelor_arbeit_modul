import pandas as pd
import networkx as nx
import sys

from ..Fine_Tuning.fine_tuning3 import *
from ..Housenumber.housenumber import *
from ..Blocker.blocker import *
from ..Feature_Extraction.feature_extraction import *


def add_tuple_column(df: pd.DataFrame):
    df["ID_TUPLE"] = df.apply(
        lambda row: (row["INDEX_SOURCE_DF1"], row["INDEX_SOURCE_DF2"]), axis=1
    )
    return df


def generate_match_ids(df: pd.DataFrame):

    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(f"df1_{row['INDEX_SOURCE_DF1']}", f"df2_{row['INDEX_SOURCE_DF2']}")

    components = list(nx.connected_components(G))

    match_ids = {}
    for match_id, component in enumerate(components, start=1):
        for node in component:
            match_ids[node] = match_id

    df["MATCH_ID"] = df.apply(
        lambda row: match_ids[f"df1_{row['INDEX_SOURCE_DF1']}"], axis=1
    )

    return df


def matcher(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    method: str = "fine-tuning",
    batch_size: int = 16,
    street_model_path: str ,
    company_model_path: str ,
):

    df1 = df1[
        (
            df1["POSTALCODE"].notnull()
            & df1["COMPANYNAME"].notnull()
            & df1["STREET"].notnull()
            & df1["HSNR"].notnull()
        )
    ]

    df2 = df2[
        (
            df2["POSTALCODE"].notnull()
            & df2["COMPANYNAME"].notnull()
            & df2["STREET"].notnull()
            & df2["HSNR"].notnull()
        )
    ]

    df1 = df1[df1["POSTALCODE"].isin(df2["POSTALCODE"])]
    df2 = df2[df2["POSTALCODE"].isin(df1["POSTALCODE"])]

    blocked_df = plz_blocking(df1, df2, "POSTALCODE")

    all_candidates = pd.DataFrame()
    for block in blocked_df.values():
        all_candidates = pd.concat([all_candidates, Indexing(block)], ignore_index=True)

    can_hsnr = hsnr_preprocessing(all_candidates, "HSNR_DF1")
    can_hsnr = hsnr_preprocessing(all_candidates, "HSNR_DF2")
    can_hsnr = compare_house_numbers(
        can_hsnr, "HSNR_DF1_CLEAN", "HSNR_DF2_CLEAN", threshold=0.75
    )

    if method == "fine-tuning":

        df1_predicted = load_model_and_predict(
            "cuda",
            street_model_path,
            can_hsnr,
            batch_size=batch_size,
            pred_street=True,
            pred_company=False,
        )
        df2_predicted = load_model_and_predict(
            "cuda",
            company_model_path,
            df1_predicted,
            batch_size=batch_size,
            pred_street=False,
            pred_company=True,
        )

        df_matched = generate_match_ids(
            df2_predicted[
                (df2_predicted["PREDICTIONS_COMPANY"] == 1)
                & (df2_predicted["PREDICTIONS_STREET"] == 1)
            ]
        )

    if method == "feature_extraction":

        df_features = extract_features(
            "cuda", can_hsnr, extract_street=True, extract_company=True, label=False
        )
        df_predict = predict_FE_both(
            df_features,
            street_model_path,
            company_model_path,
            predict_street=True,
            predict_company=True,
        )

        df_matched = generate_match_ids(
            df_predict[
                (df_predict["PREDICTIONS_STREET"].isin({1, "1"}))
                & (df_predict["PREDICTIONS_COMPANY"].isin({1, "1"}))
            ]
        )

    return df_matched.drop(columns={"STREET_SERIALIZED", "COMPANY_SERIALIZED"})