import pandas as pd
import os
import torch
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import classification_report
from typing import List
import joblib
import sys
from sklearn.model_selection import StratifiedKFold
import matplotlib as plt

################## Confusion Matrix #########################################
def Confusion_matrix(labels, predictions) -> None:
    conf_matrix = confusion_matrix(labels, predictions)

    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, pos_label=1)
    recall = recall_score(labels, predictions, pos_label=1)
    f1 = f1_score(labels, predictions, pos_label=1)

    specificity = TN / (TN + FP)
    print(f"Confusion Matrix: ")
    print(conf_matrix)
    print(f"accuracy : {accuracy}")
    print(f"precision : {precision}")
    print(f"recall : {recall}")
    print(f"f1 : {f1}")
    print(f"specificity : {specificity}")
    return


############################# Confusion Matrix Heatmap ######################################


def calculate_and_plot_heatmap(
    df: pd.DataFrame, label_col: str, prediction_col: str, similarity_col: str
):

    similarity_ranges = [
        ("60% <= x < 70%", 0.6, 0.7),
        ("70% <= x < 80%", 0.7, 0.8),
        ("80% <= x < 90%", 0.8, 0.9),
        ("90% <= x < 100%", 0.9, 1.0),
        ("x = 100%", 1.0, 1.1),
    ]

    counts = {rng[0]: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for rng in similarity_ranges}

    for label, lower_bound, upper_bound in similarity_ranges:
        subset = df[
            (df[similarity_col] >= lower_bound) & (df[similarity_col] < upper_bound)
        ]
        tp = ((subset[label_col] == 1) & (subset[prediction_col] == 1)).sum()
        tn = ((subset[label_col] == 0) & (subset[prediction_col] == 0)).sum()
        fp = ((subset[label_col] == 0) & (subset[prediction_col] == 1)).sum()
        fn = ((subset[label_col] == 1) & (subset[prediction_col] == 0)).sum()

        counts[label] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    count_df = pd.DataFrame(counts).T

    count_df = count_df[(count_df.sum(axis=1) > 0)]

    plt.figure(figsize=(12, 6))
    sns.heatmap(count_df, annot=True, cmap="YlGnBu", fmt="d", linewidths=0.5)
    plt.title("Konfusionsmatrix unter Berücksichtigung der Ähnlichkeitswerte")
    plt.show()


############################### Feature Extraction ############################################


def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output[0]
    # Expanding attention mask  -> token embeddings dimensions
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def extract_features_final(
    device,
    data: pd.DataFrame,
    extract_street: bool = True,
    extract_company: bool = True,
    label: bool = False,
    pooling_type: str = "cls",
) -> pd.DataFrame:

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-german-cased")
    model = DistilBertModel.from_pretrained("distilbert-base-german-cased").to(device)

    street_cache = {}
    company_cache = {}

    def compute_features(row):
        def apply_pooling(outputs, attention_mask, pooling_type):
            if pooling_type == "cls":
                return (
                    outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().flatten()
                )
            elif pooling_type == "mean":
                return (
                    mean_pooling(outputs, attention_mask)
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )
            elif pooling_type == "max":
                max_pooling, _ = torch.max(outputs.last_hidden_state, dim=1)
                return max_pooling.detach().cpu().numpy().flatten()
            else:
                raise ValueError(f"Unsupported pooling type: {pooling_type}")

        # Street Features
        if extract_street:
            street_key = row["STREET_DF1"] + " [SEP] " + row["STREET_DF2"]
            if street_key not in street_cache:
                inputs_street = tokenizer(
                    street_key, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs_street = model(**inputs_street)
                pooled_output_street = apply_pooling(
                    outputs_street, inputs_street["attention_mask"], pooling_type
                )
                street_cache[street_key] = pooled_output_street
            row["FEATURES_STREET"] = street_cache[street_key]

            if label:
                row["LABELS_STREET"] = 1 if row["LABELS_STREET"] == "Match" else 0

        # Company Features
        if extract_company:
            company_key = row["COMPANYNAME_DF1"] + " [SEP] " + row["COMPANYNAME_DF2"]
            if company_key not in company_cache:
                inputs_company = tokenizer(
                    company_key, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs_company = model(**inputs_company)
                pooled_output_company = apply_pooling(
                    outputs_company, inputs_company["attention_mask"], pooling_type
                )
                company_cache[company_key] = pooled_output_company
            row["FEATURES_COMPANY"] = company_cache[company_key]

            if label:
                row["LABELS_COMPANY"] = 1 if row["LABELS_COMPANY"] == "Match" else 0

        return row

    data = data.apply(compute_features, axis=1)
    return data


####################### Train Classifier ######################################


def train_CF(
    input_model: str = None,
    features: list = None,
    labels: list = None,
    output_model: str = None,
):
    if input_model is None:
        classifier = LogisticRegression(max_iter=200)
    else:
        classifier = joblib.load(input_model)

    classifier.fit(features, labels)
    joblib.dump(classifier, output_model)

    return None


#################### Prediction ##############################


def predict_FE_both(
    df: pd.DataFrame,
    street_model_path=None,
    company_model_path=None,
    predict_street: bool = True,
    predict_company: bool = True,
) -> pd.DataFrame:
    if predict_street:

        classifier_street = joblib.load(street_model_path)
        df["PREDICTIONS_STREET"] = classifier_street.predict(
            list(df["FEATURES_STREET"])
        )

    if predict_company:

        classifier_company = joblib.load(company_model_path)
        df["PREDICTIONS_COMPANY"] = classifier_company.predict(
            list(df["FEATURES_COMPANY"])
        )

    return df


############## Cross Validation ###############################


def cross_validate_fe(
    df: pd.DataFrame,
    stratify_col: str,
    device,
    output_dir: str,
    k: int = 4,
    train_street: bool = True,
    train_company: bool = False,
    pooling_type: str = "cls",
):
    stratify_labels = df[stratify_col].astype(str).str[:3]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = {}

    df = extract_features_final(
        device,
        df,
        extract_street=train_street,
        extract_company=train_company,
        label=True,
        pooling_type=pooling_type,
    )
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, stratify_labels)):
        print(f"Training fold {fold + 1}/{k}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        print(f"Training size : {train_df.shape[0]}")
        print(f"Test size : {val_df.shape[0]}")

        if train_street:
            train_features_street = list(train_df["FEATURES_STREET"])
            train_labels_street = list(train_df["LABELS_STREET"])
            val_features_street = list(val_df["FEATURES_STREET"])
            val_labels_street = list(val_df["LABELS_STREET"])

            street_model_path = os.path.join(
                output_dir, f"fold_{fold + 1}_street_model.joblib"
            )
            train_CF(
                features=train_features_street,
                labels=train_labels_street,
                output_model=street_model_path,
                input_model=None,
            )
            val_df = predict_FE_both(
                val_df,
                street_model_path=street_model_path,
                predict_street=True,
                predict_company=False,
            )
            Confusion_matrix(
                list(val_df["LABELS_STREET"]), list(val_df["PREDICTIONS_STREET"])
            )

        if train_company:
            train_features_company = list(train_df["FEATURES_COMPANY"])
            train_labels_company = list(train_df["LABELS_COMPANY"])
            val_features_company = list(val_df["FEATURES_COMPANY"])
            val_labels_company = list(val_df["LABELS_COMPANY"])

            company_model_path = os.path.join(
                output_dir, f"fold_{fold + 1}_company_model.joblib"
            )
            train_CF(
                features=train_features_company,
                labels=train_labels_company,
                output_model=company_model_path,
                input_model=None,
            )
            val_df = predict_FE_both(
                val_df,
                company_model_path=company_model_path,
                predict_street=False,
                predict_company=True,
            )
            Confusion_matrix(
                list(val_df["LABELS_COMPANY"]), list(val_df["PREDICTIONS_COMPANY"])
            )

        fold_results[fold] = {"df_val": val_df}

    return fold_results
