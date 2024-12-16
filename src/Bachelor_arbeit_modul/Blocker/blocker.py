import pandas as pd
import math
import numpy as np
import torch


############  plz_blocking  ############################
def remove_suffix(col_name, suffix: str):

    if col_name.endswith(suffix):
        return col_name[: -len(suffix)]

    return col_name


def plz_blocking(df1, df2, column: str):

    df1.rename(columns=lambda x: remove_suffix(x, "_DF1"), inplace=True)
    df2.rename(columns=lambda x: remove_suffix(x, "_DF2"), inplace=True)

    df1["SOURCE"] = "DF1"

    df2["SOURCE"] = "DF2"

    combined_df = pd.concat([df1, df2], ignore_index=True)
    blocked_dfs = {
        block_key: block_df for block_key, block_df in combined_df.groupby(column)
    }
    return blocked_dfs


############ dynamic Window Slicing ############################


def window(a: int, b: int) -> int:
    def count_digits(n: int) -> int:
        return len(str(n))

    s = a + b

    if a > b:
        r = a // b
    else:
        r = b // a

    dig_r = count_digits(r)

    if dig_r >= 2:
        window_size = max(2, min(r + 1, 51))
    elif dig_r <= 1:
        window_size = max(2, min(int(s * 0.60), 31))

    if window_size % 2 == 0 and s != 2:
        window_size += 1

    return window_size


############# sorted Neighbourhood Indexing ##################
def Indexing(block: pd.DataFrame) -> pd.DataFrame:

    comparable_tuples = pd.DataFrame()
    df1 = block[block["SOURCE"] == "DF1"]
    df2 = block[block["SOURCE"] == "DF2"]

    df1_size = df1.shape[0]
    df2_size = df2.shape[0]

    windowSize = window(df1_size, df2_size)

    df1["INDEX_SOURCE_DF1"] = df1.index.astype(str)
    df2["INDEX_SOURCE_DF2"] = df2.index.astype(str)

    df_all = (
        pd.concat([df1, df2])
        .sort_values(["STREET", "HSNR", "COMPANYNAME"])
        .reset_index(drop=True)
    )
    df_all["INDEX_ALL"] = df_all.index

    if windowSize == 2:
        df1 = df1.add_suffix("_DF1")
        df2 = df2.add_suffix("_DF2")
        comparable_dfs = (
            df1.reset_index(drop=True)
            .merge(
                df2[
                    [
                        "INDEX_SOURCE_DF2_DF2",
                        "COMPANYNAME_DF2",
                        "POSTALCODE_DF2",
                        "CITY_DF2",
                        "STREET_DF2",
                        "HSNR_DF2",
                    ]
                ].reset_index(drop=True),
                right_index=True,
                left_index=True,
            )
            .reset_index(drop=True)
        )

        comparable_dfs = comparable_dfs.rename(
            columns={
                "INDEX_SOURCE_DF1_DF1": "INDEX_SOURCE_DF1",
                "INDEX_SOURCE_DF2_DF2": "INDEX_SOURCE_DF2",
            }
        )[
            [
                "COMPANYNAME_DF1",
                "POSTALCODE_DF1",
                "CITY_DF1",
                "STREET_DF1",
                "HSNR_DF1",
                "INDEX_SOURCE_DF1",
                "INDEX_SOURCE_DF2",
                "COMPANYNAME_DF2",
                "POSTALCODE_DF2",
                "CITY_DF2",
                "STREET_DF2",
                "HSNR_DF2",
            ]
        ]

        return comparable_dfs.reset_index(drop=True)

    else:

        tuples_wnd = []
        base_indices = (
            df_all.loc[~df_all["INDEX_SOURCE_DF1"].isnull()]["INDEX_ALL"]
            if df1_size < df2_size
            else df_all.loc[~df_all["INDEX_SOURCE_DF2"].isnull()]["INDEX_ALL"]
        ).values

        wsize_half = max(1, int((windowSize - 1) / 2))

        max_dist_tolerance = max(2, int(wsize_half / 6))

        bi_pre = math.inf
        for b in range(len(base_indices)):

            bi = base_indices[b]

            if abs(bi - bi_pre) > max_dist_tolerance:

                bi_pre = bi
                window_indices = list(range((bi - wsize_half), (bi + wsize_half) + 1))
                for i in window_indices:
                    for j in window_indices:
                        if j >= i:
                            tuples_wnd.append([i, j])

        tuples_wnd = np.unique(tuples_wnd, axis=0)

        tuples_df = pd.DataFrame(
            data={"I_FU1": tuples_wnd[:, 0], "I_FU2": tuples_wnd[:, 1]}
        )

        tuples_df[:] = tuples_df[:].astype(int)

        # Valid Tuples
        comparable_tuples = pd.DataFrame()
        fusion_lookup = df_all[["INDEX_SOURCE_DF1", "INDEX_SOURCE_DF2", "INDEX_ALL"]]
        translated_tuples = (
            tuples_df.merge(
                fusion_lookup.rename(columns={"INDEX_ALL": "I_FU1"}),
                on="I_FU1",
                how="left",
            )
            .rename(
                columns={
                    "INDEX_SOURCE_DF1": "INDEX_DF1_FU1",
                    "INDEX_SOURCE_DF2": "INDEX_DF2_FU1",
                }
            )
            .merge(
                fusion_lookup.rename(columns={"INDEX_ALL": "I_FU2"}),
                on="I_FU2",
                how="left",
            )
            .rename(
                columns={
                    "INDEX_SOURCE_DF1": "INDEX_DF1_FU2",
                    "INDEX_SOURCE_DF2": "INDEX_DF2_FU2",
                }
            )[
                [
                    "INDEX_DF1_FU1",
                    "INDEX_DF2_FU1",
                    "INDEX_DF1_FU2",
                    "INDEX_DF2_FU2",
                ]
            ]
        )
        comparable_tuples_nh = translated_tuples.loc[
            (
                ~translated_tuples["INDEX_DF1_FU1"].isnull()
                & ~translated_tuples["INDEX_DF2_FU2"].isnull()
            )
            | (
                ~translated_tuples["INDEX_DF2_FU1"].isnull()
                & ~translated_tuples["INDEX_DF1_FU2"].isnull()
            )
        ]
        comparable_tuples = pd.concat(
            [comparable_tuples, comparable_tuples_nh], axis=0
        ).drop_duplicates()

        comparable_tuples.insert(len(comparable_tuples.columns), "DF1", np.nan)

        comparable_tuples.insert(len(comparable_tuples.columns), "DF2", np.nan)

        for df in ["DF1", "DF2"]:

            for fu in ["FU1", "FU2"]:

                comparable_tuples.loc[
                    ~comparable_tuples[f"INDEX_{df}_{fu}"].isnull(), df
                ] = comparable_tuples.loc[
                    ~comparable_tuples[f"INDEX_{df}_{fu}"].isnull()
                ][
                    f"INDEX_{df}_{fu}"
                ]

        # Comparable dfs
        comparable_tuples = comparable_tuples[["DF1", "DF2"]]
        comparable_tuples["DF1"] = comparable_tuples["DF1"].astype(int)
        comparable_tuples["DF2"] = comparable_tuples["DF2"].astype(int)
        df1 = df1.add_suffix("_DF1")
        df2 = df2.add_suffix("_DF2")
        comparable_dfs = (
            comparable_tuples.merge(df1, left_on="DF1", right_index=True)
            .merge(df2, left_on="DF2", right_index=True)
            .reset_index(drop=True)
        )
        comparable_dfs = comparable_dfs.rename(
            columns={
                "INDEX_SOURCE_DF1_DF1": "INDEX_SOURCE_DF1",
                "INDEX_SOURCE_DF2_DF2": "INDEX_SOURCE_DF2",
            }
        )[
            [
                "COMPANYNAME_DF1",
                "POSTALCODE_DF1",
                "CITY_DF1",
                "STREET_DF1",
                "HSNR_DF1",
                "INDEX_SOURCE_DF1",
                "INDEX_SOURCE_DF2",
                "COMPANYNAME_DF2",
                "POSTALCODE_DF2",
                "CITY_DF2",
                "STREET_DF2",
                "HSNR_DF2",
            ]
        ]

        return comparable_dfs
