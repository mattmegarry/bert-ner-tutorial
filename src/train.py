import pandas as pd
import numpy as np

import torch
import joblib

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel


def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    pos = df.groupby("Sentence #")["POS"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values

    return sentences, pos, tag, enc_pos, enc_tag

# read data in
if __name__ == "__main__": # whats this in python?
    sentences, pos, tag, enc_pos, enc_tag = process_data(config.TRAINING_FILE)
    
    meta_data = {
        "enc_pos": enc_pos,
        "enc_tag": enc_tag
    }
    
    joblib.dump(meta_data, "meta.bin") # "You could dump with the config if you want". "Saving this will give me the encoders for doing inference later".

    num_pos = len(list(enc_pos._classes)) # "The number of unique POS tags in the dataset" - should explore how that was generated from the label encoder
    num_tag = len(list(enc_tag._classes)) # "_classes gives yo uall the different classes" - in a distinct/uniques way or they already are? ._something looks like a convention for something 

    (
        train_sentences,
        test_sentences,
        train_pos,
        test_pos,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, pos, tag, random_state=42, test_size=0.1) # "Small test size since there's not a lot of data"
# Is it the case that if you just returns n*2 lists where n is however many datasets you gave it? (in this case 3?)

