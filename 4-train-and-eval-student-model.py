import os
import argparse
import pandas as pd
import json
import numpy as np
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("current_version", help="for instance v1.")
    args = parser.parse_args()

current_version = args.current_version

print(current_version)

# Open training data
train_df = pd.read_json("datasets/X-GENRE-train-with-LLM-predictions.jsonl", orient="records", lines=True)

# Change the train df so that you only have text and GPT labels
train_df = train_df[["text", "LLM_label"]]

train_df.rename(columns={"LLM_label": "labels"}, inplace=True)

print(f"Training data opened. Training data size: {train_df.shape}")

# Create a list of labels
LABELS = train_df.labels.unique().tolist()
print(LABELS)

output_dir = f"models/Student-X-GENRE-{current_version}"

model_args = ClassificationArgs()

model_args={
    "overwrite_output_dir": True,
    "output_dir": output_dir,
    "num_train_epochs": 15,
    "train_batch_size":8,
    "learning_rate": 1e-5,
    "labels_list": LABELS,
    "max_seq_length": 512,
    "save_steps": -1,
    # Only the trained model will be saved - to prevent filling all of the space
    "save_model_every_epoch":False,
    "silent": True,
    "use_multiprocessing":False,
    "use_multiprocessing_for_evaluation":False,
    }

# Create a TransformerModel - use the same hyperparameters as the original X-GENRE model
roberta_base_model = ClassificationModel(
        "xlmroberta", "xlm-roberta-base",
        num_labels=len(LABELS),
        use_cuda=True,
        args= model_args
        )

print("Training started.")

# Train the model on train data
roberta_base_model.train_model(train_df)

print("Training completed.")