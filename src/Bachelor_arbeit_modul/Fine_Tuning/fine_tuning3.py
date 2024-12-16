import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    DistilBertForSequenceClassification,
    DistilBertPreTrainedModel,
    DistilBertConfig,
    DistilBertConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score
from transformers import AdamW
from sklearn.model_selection import StratifiedKFold
from ..Feature_Extraction.feature_extraction import *

############ Help Functions ###################


def proportional_sample(df: pd.DataFrame, column: str, n: int):

    proportions = df[column].astype(str).str[:3].value_counts(normalize=True)

    samples = []
    for value, proportion in proportions.items():
        sample_size = int(n * proportion)
        sample = df[df[column].astype(str).str[:3] == value].sample(
            sample_size, random_state=42
        )
        samples.append(sample)

    return pd.concat(samples)


def custom_data_collator(features):
    input_ids = torch.stack([feature[0] for feature in features])
    attention_masks = torch.stack([feature[1] for feature in features])
    labels = torch.tensor([feature[2] for feature in features])
    return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}


############ Serialization & Tokenization ######


def serialization(
    df: pd.DataFrame,
    street_ser: bool = True,
    company_ser: bool = True,
    label: bool = False,
) -> pd.DataFrame:

    if street_ser:
        df["STREET_DF1"] = df["STREET_DF1"].str.strip()
        df["STREET_DF2"] = df["STREET_DF2"].str.strip()
        df["STREET_SERIALIZED"] = df["STREET_DF1"] + " [SEP] " + df["STREET_DF2"]
        if label:
            df["LABELS_STREET"].replace({"Match": 1, "Not Match": 0}, inplace=True)

    if company_ser:
        df["COMPANYNAME_DF1"] = df["COMPANYNAME_DF1"].str.strip()
        df["COMPANYNAME_DF2"] = df["COMPANYNAME_DF2"].str.strip()
        df["COMPANY_SERIALIZED"] = (
            df["COMPANYNAME_DF1"] + " [SEP] " + df["COMPANYNAME_DF2"]
        )
        if label:
            df["LABELS_COMPANY"].replace({"Match": 1, "Not Match": 0}, inplace=True)

    return df


def tokenization(
    df: pd.DataFrame,
    tokenizer,
    tok_street: bool = False,
    tok_company: bool = False,
    label: bool = False,
):
    if tok_street:
        street_tokenized = tokenizer(
            df["STREET_SERIALIZED"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if label:

            labels_tensor = torch.tensor(df["LABELS_STREET"].tolist())
            tokenized_dataset = TensorDataset(
                street_tokenized["input_ids"].cpu(),
                street_tokenized["attention_mask"].cpu(),
                labels_tensor.cpu(),
            )
        else:
            tokenized_dataset = TensorDataset(
                street_tokenized["input_ids"].cpu(),
                street_tokenized["attention_mask"].cpu(),
            )

    if tok_company:
        company_tokenized = tokenizer(
            df["COMPANY_SERIALIZED"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if label:
            labels_tensor = torch.tensor(df["LABELS_COMPANY"].tolist())
            tokenized_dataset = TensorDataset(
                company_tokenized["input_ids"].cpu(),
                company_tokenized["attention_mask"].cpu(),
                labels_tensor.cpu(),
            )
        else:
            tokenized_dataset = TensorDataset(
                company_tokenized["input_ids"].cpu(),
                company_tokenized["attention_mask"].cpu(),
            )

    return tokenized_dataset


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class CustomDistilBertConfig(DistilBertConfig):
    def __init__(
        self,
        activation_function=None,
        seq_classif_dropout: float = 0.2,
        activation_position: str = "before",
        without_activation_function: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation_function = activation_function
        self.seq_classif_dropout = seq_classif_dropout
        self.activation_position = activation_position
        self.without_activation_function = without_activation_function


class DistilBertForCustomClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, self.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.activation = None
        if config.activation_function == "relu":
            self.activation = nn.ReLU()
        elif config.activation_function == "sigmoid":
            self.activation = nn.Sigmoid()
        elif config.activation_function == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif config.activation_function == "tanh":
            self.activation = nn.Tanh()
        elif config.activation_function is None:
            self.activation = None
        else:
            raise ValueError(
                f"Unsupported activation function: {config.activation_function}"
            )

        self.activation_position = config.activation_position
        self.without_activation_function = config.without_activation_function

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)

        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        if self.activation is not None and self.activation_position == "before":
            pooled_output = self.activation(pooled_output)

        logits = self.classifier(pooled_output)  # (bs, num_labels)

        if self.activation is not None and self.activation_position == "after":
            logits = self.activation(logits)

        return SequenceClassifierOutput(logits=logits)


def train_and_evaluate(
    device,
    train_dataset,
    val_dataset,
    val_df,
    activation_function=None,
    dropout: float = 0.2,
    activation_position: str = "before",
    without_activation_function: bool = False,
    optimizer_name: str = "adamw",
    output_dir: str = "./outputs",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 5e-5,
    logging_dir: str = "./logs",
    train_street: bool = True,
    train_company: bool = False,
    weight_decay: float = 0.01,
) -> pd.DataFrame:

    config = CustomDistilBertConfig.from_pretrained(
        "distilbert-base-german-cased",
        activation_function=activation_function,
        seq_classif_dropout=dropout,
        activation_position=activation_position,
        without_activation_function=without_activation_function,
        num_labels=2,
    )

    model = DistilBertForCustomClassification.from_pretrained(
        "distilbert-base-german-cased", config=config
    ).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_dir=logging_dir,
        logging_steps=100,
        save_steps=100,
        eval_steps=100,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="tensorboard",
        weight_decay=weight_decay,
    )

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),  # Only optimizer is passed, scheduler is None
    )

    train_output = trainer.train()

    global_step = train_output.global_step
    training_loss = train_output.training_loss
    metrics = train_output.metrics

    print(f"Total training steps: {global_step}")
    print(f"Final training loss: {training_loss}")
    print("Training metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    eval_metrics = trainer.evaluate()

    print("Evaluation metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")

    predictions = trainer.predict(val_dataset)

    if train_street:
        val_df["PREDICTIONS_STREET"] = predictions.predictions.argmax(axis=-1)
        val_df["TRUE_LABELS_STREET"] = predictions.label_ids
        print(
            Confusion_matrix(val_df["TRUE_LABELS_STREET"], val_df["PREDICTIONS_STREET"])
        )

    if train_company:
        val_df["PREDICTIONS_COMPANY"] = predictions.predictions.argmax(axis=-1)
        val_df["TRUE_LABELS_COMPANY"] = predictions.label_ids
        print(
            Confusion_matrix(
                val_df["TRUE_LABELS_COMPANY"], val_df["PREDICTIONS_COMPANY"]
            )
        )

    model_save_path = os.path.join(output_dir, "model")
    model.save_pretrained(model_save_path)

    return val_df


################# Prediction Function #############################


def load_model_and_predict(
    device,
    model_path: str,
    df: pd.DataFrame,
    batch_size: int = 32,
    pred_street: bool = False,
    pred_company: bool = False,
) -> pd.DataFrame:
    predictions = []
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-german-cased")
    model = DistilBertForCustomClassification.from_pretrained(model_path)
    model.to(device)

    if pred_street:
        serialized = serialization(df, street_ser=True, company_ser=False, label=False)
        tokenized = tokenization(
            serialized, tokenizer, tok_street=True, tok_company=False, label=False
        )

    elif pred_company:
        serialized = serialization(df, street_ser=False, company_ser=True, label=False)
        tokenized = tokenization(
            serialized, tokenizer, tok_street=False, tok_company=True, label=False
        )

    model.eval()
    dataloader = DataLoader(
        tokenized, sampler=SequentialSampler(tokenized), batch_size=batch_size
    )
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    if pred_street:
        df["PREDICTIONS_STREET"] = predictions
    elif pred_company:
        df["PREDICTIONS_COMPANY"] = predictions

    return df


################## Cross Validation ################################


def cross_validate(
    df: pd.DataFrame,
    stratify_col: str,
    output_dir: str,
    logging_dir: str,
    device,
    dropout: float,
    without_activation_function: bool = False,
    k: int = 4,
    train_street: bool = True,
    train_company: bool = False,
    **kwargs,
):
    if stratify_col in {"STREET_WORDDIFF_LOCALITY", "COMPANYNAME_WORDDIFF"}:

        stratify_labels = df[stratify_col].astype(str).str[:3]

    else:
        stratify_labels = df[stratify_col]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, stratify_labels)):
        print(f"Training fold {fold + 1}/{k}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-german-cased")

        if train_street:
            train_serialized = serialization(
                train_df, street_ser=True, company_ser=False, label=True
            )
            train_dataset = tokenization(
                train_serialized,
                tokenizer,
                tok_street=True,
                tok_company=False,
                label=True,
            )

            val_serialized = serialization(
                val_df, street_ser=True, company_ser=False, label=True
            )
            val_dataset = tokenization(
                val_serialized,
                tokenizer,
                tok_street=True,
                tok_company=False,
                label=True,
            )

        if train_company:
            train_serialized = serialization(
                train_df, street_ser=False, company_ser=True, label=True
            )
            train_dataset = tokenization(
                train_serialized,
                tokenizer,
                tok_street=False,
                tok_company=True,
                label=True,
            )

            val_serialized = serialization(
                val_df, street_ser=False, company_ser=True, label=True
            )
            val_dataset = tokenization(
                val_serialized,
                tokenizer,
                tok_street=False,
                tok_company=True,
                label=True,
            )

        fold_output_dir = os.path.join(output_dir, f"fold_{fold + 1}")
        fold_logging_dir = os.path.join(logging_dir, f"fold_{fold + 1}")

        os.makedirs(fold_output_dir, exist_ok=True)
        os.makedirs(fold_logging_dir, exist_ok=True)

        val_df = train_and_evaluate(
            device,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            val_df=val_df,
            output_dir=fold_output_dir,
            logging_dir=fold_logging_dir,
            train_street=train_street,
            train_company=train_company,
            dropout=dropout,
            without_activation_function=without_activation_function,
            **kwargs,
        )

        fold_results[fold] = {"df_val": val_df}

    return fold_results


######################## Evaluation Function ################################


def evaluation(
    model,
    tokenizer,
    batch_size,
    df: pd.DataFrame,
    eval_street: bool = False,
    eval_company: bool = False,
) -> pd.DataFrame:
    predictions = []
    true_labels = []

    if eval_street:
        serialized = serialization(df, street_ser=True, company_ser=False, label=True)
        tokenized = tokenization(
            serialized, tokenizer, tok_street=True, tok_company=False, label=True
        )

    elif eval_company:
        serialized = serialization(df, street_ser=False, company_ser=True, label=True)
        tokenized = tokenization(
            serialized, tokenizer, tok_street=False, tok_company=True, label=True
        )

    dataloader = DataLoader(
        tokenized, sampler=SequentialSampler(tokenized), batch_size=batch_size
    )
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            label_ids = labels.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(label_ids)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    if eval_street:
        df["PREDICTIONS_STREET"] = predictions
    elif eval_company:
        df["PREDICTIONS_COMPANY"] = predictions

    print(Confusion_matrix(true_labels, predictions))
    torch.cuda.empty_cache()

    return df
