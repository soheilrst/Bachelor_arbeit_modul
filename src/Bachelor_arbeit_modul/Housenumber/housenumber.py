import pandas as pd
import math
import numpy as np
import re


def hsnr_preprocessing(df: pd.DataFrame, column: str) -> pd.DataFrame:

    column_clean = column + "_CLEAN"
    df[column_clean] = df[column]
    df[column_clean].fillna("", inplace=True)
    df[column_clean] = df[column_clean].str.strip()
    df[column_clean] = df[column_clean].str.lower()

    # extra zeros type1 0000
    pattern_extra_zero = r"^0{2,}$"
    df[column_clean] = df[column_clean].str.replace(pattern_extra_zero, "0", regex=True)

    # extra zeros  type2  005
    pattern_extra_zero = r"^(0+)(\d+)"
    df[column_clean] = df[column_clean].str.replace(
        pattern_extra_zero, r"\2", regex=True
    )

    # nr at the beginnig should be removed
    pattern_extra_nr = "^(nr\.{0,1})(\d+)"
    df[column_clean] = df[column_clean].str.replace(pattern_extra_nr, r"\2", regex=True)

    # if the housbumber is just special characters -> should be removed
    df[column_clean] = np.where(
        df[column_clean].str.contains("^\W+$", regex=True, na=False),
        df[column_clean].str.replace("^\W+$", "", regex=True),
        df[column_clean],
    )

    # unbekannt word should be removed
    df[column_clean] = np.where(
        df[column_clean].str.contains(r"(?i)\s*unbekannt\s*", regex=True, na=False),
        df[column_clean].str.replace("\s*unbekannt\s*", "", regex=True),
        df[column_clean],
    )
    return df


def extract_multiple_numbers_sets(s):
    return (
        list(map(int, re.findall(r"\d+", s)[1:]))
        if len(re.findall(r"\d+", s)) > 1
        else []
    )


def extract_first_number_set(s):
    match = re.search(r"\d+", s)
    return [int(match.group(0))] if match else []


def extract_letters(s):
    return re.findall(r"[a-zA-ZäÄöÖüÜß]{1,2}", s)


def extract_words(s):
    return re.findall(r"[a-zA-ZäÄöÖüÜß]{3,}", s)


# for comparing the first number with first number and letter with letter
def compare_components(comp1, comp2):

    # both of the columns are empty
    if not comp1 and not comp2:
        return 1

    # one of the columns is empty
    elif not comp1 or not comp2:
        return 0.0

    # if the columns are the same
    elif (comp1 and comp2) and (set(comp1) == set(comp2)):
        return 1.0

    # if they have some values in common
    elif (comp1 and comp2) and (
        set(comp1).issubset(set(comp2)) or set(comp2).issubset(set(comp1))
    ):
        return 0.75

    else:
        return 0.0


# for comparing the different kind of columns with eachother , first column with range
def compare_components2(comp1, comp2):

    # if one of the columns is empty
    if not comp2:
        return 1

    # if the columns are the same
    elif (comp1 and comp2) and (set(comp1) == set(comp2)):
        return 1.0

    # if they have some values in common
    elif (comp1 and comp2) and (
        set(comp1).issubset(set(comp2)) or set(comp2).issubset(set(comp1))
    ):
        return 0.75

    else:
        return 0


def calculate_similarity(row):

    numbers_score = (
        compare_components(row["HSNR_FIRST_1"], row["HSNR_FIRST_2"]) * 0.8
        + compare_components2(row["HSNR_RANGE_1"], row["HSNR_RANGE_2"]) * 0.2
        # compare_components2(row["HSNR_FIRST_1"], row["HSNR_RANGE_2"])*.1 +
        # compare_components2(row["HSNR_FIRST_2"], row["HSNR_RANGE_1"])*.1
    )
    word_score = compare_components2(row["HSNR_WORD_1"], row["HSNR_WORD_2"])
    letters_score = compare_components(row["HSNR_LETTER_1"], row["HSNR_LETTER_2"])
    score = numbers_score * 0.65 + letters_score * 0.25 + word_score * 0.1
    return score


def compare_house_numbers(
    df: pd.DataFrame, column1: str, column2: str, threshold: float = None
):

    # create help columns
    df["HSNR_FIRST_1"] = ""
    df["HSNR_RANGE_1"] = ""
    df["HSNR_LETTER_1"] = ""
    df["HSNR_WORD_1"] = ""

    df["HSNR_FIRST_2"] = ""
    df["HSNR_RANGE_2"] = ""
    df["HSNR_LETTER_2"] = ""
    df["HSNR_WORD_2"] = ""

    # just the row with hsnr
    df = df[(df[column1] != "") & (df[column2] != "")]

    # Extract components for column1
    df["HSNR_FIRST_1"] = df[column1].apply(extract_first_number_set)
    df["HSNR_RANGE_1"] = df[column1].apply(extract_multiple_numbers_sets)
    df["HSNR_WORD_1"] = np.where(
        df[column1].str.contains(r"[a-zA-ZäÄöÖüÜß]{3,}", regex=True),
        df[column1].apply(extract_words),
        df["HSNR_WORD_1"],
    )

    df["HSNR_LETTER_1"] = np.where(
        (
            df[column1].str.contains(r"[a-zA-ZäÄöÖüÜß]{1,2}", regex=True)
            & (df["HSNR_WORD_1"].apply(len) == 0)
        ),
        df[column1].apply(extract_letters),
        df["HSNR_LETTER_1"],
    )

    # Extract components for column2
    df["HSNR_FIRST_2"] = df[column2].apply(extract_first_number_set)
    df["HSNR_RANGE_2"] = df[column2].apply(extract_multiple_numbers_sets)
    df["HSNR_WORD_2"] = np.where(
        df[column2].str.contains(r"[a-zA-ZäÄöÖüÜß]{3,}", regex=True),
        df[column2].apply(extract_words),
        df["HSNR_WORD_2"],
    )

    df["HSNR_LETTER_2"] = np.where(
        (
            df[column2].str.contains(r"[a-zA-ZäÄöÖüÜß]{1,2}", regex=True)
            & (df["HSNR_WORD_2"].apply(len) == 0)
        ),
        df[column2].apply(extract_letters),
        df["HSNR_LETTER_2"],
    )

    # calculate similarity of the components
    df["HSNR_SIM"] = df.apply(calculate_similarity, axis=1)
    df["HSNR_SIM"] = df["HSNR_SIM"].round(2)
    df.drop(
        columns=[
            "HSNR_DF1_CLEAN",
            "HSNR_DF2_CLEAN",
            "HSNR_FIRST_1",
            "HSNR_RANGE_1",
            "HSNR_LETTER_1",
            "HSNR_WORD_1",
            "HSNR_FIRST_2",
            "HSNR_RANGE_2",
            "HSNR_LETTER_2",
            "HSNR_WORD_2",
        ],
        inplace=True,
    )
    if threshold is not None:

        return df[df["HSNR_SIM"] > threshold]
    else:
        return df
