import pandas as pd
import torch
from torch.utils import data
import random
import nltk
from nltk.corpus import wordnet
from nltk import wordpunct_tokenize

random.seed(42)
nltk.download('wordnet')


def get_synonim(text):
  syn_list = []
  if "/" in text:
    text_list = text.split("/")
    text = random.choice(text_list)

  for syn in wordnet.synsets(text):
      for name in syn.lemma_names():
          syn_list.append(name.replace("_", " ").lower())

  syn_list = list(set(syn_list))
  return syn_list


# function for NLI dataset preparation
def prepare_nli_dataset(df_nli, premise_column, hypothesis_column, label_column, label_encode_dict):
    df_nli = df_nli[[label_column, premise_column, hypothesis_column]]
    df_nli.replace({label_column: label_encode_dict}, inplace=True)
    df_nli.columns = ["label", "premise", "hypothesis"]
    return df_nli

# function for preparing NLI dataset from classification dataset
def prepare_nli_dataset_from_classification(
    df_clf,
    text_column,
    label_column,
    label_template="This example is about",
    random_contradict=True,
    add_label_variation=False
):
    entails_data = []
    contradict_data = []
    labels = df_clf[label_column].unique()
    for text_item, label_item in zip(df_clf[text_column], df_clf[label_column]):
        for label in labels:
            if label == label_item:
                if add_label_variation:
                    try:
                        label_syn = get_synonim(label)
                        label = random.choice(label_syn)
                    except:
                        pass

                item = {
                    "label": 0,
                    "premise": text_item,
                    "hypothesis": f"{label_template} {label.lower()}"
                }
                entails_data.append(item)
            else:
                if add_label_variation:
                    try:
                        label_syn = get_synonim(label)
                        label = random.choice(label_syn)
                    except:
                        pass
                
                item = {
                    "label": 1,
                    "premise": text_item,
                    "hypothesis": f"{label_template} {label.lower()}"
                }
                contradict_data.append(item)

    if random_contradict:
        contradict_data = random.sample(contradict_data, len(entails_data))
    
    all_data = entails_data + contradict_data
    random.shuffle(all_data)
    return pd.DataFrame(all_data)

# Pytorch dataset object pipeline
class NliDataset(data.Dataset):
    def __init__(self, df_nli, tokenizer, max_length):
        self.df_nli = df_nli
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df_nli)

    def __getitem__(self, idx):
        label_id, premise, hypothesis = self.df_nli.values[idx]
        premise = " ".join(wordpunct_tokenize(premise)[:self.max_length//2])
        hypothesis = " ".join(wordpunct_tokenize(hypothesis)[:self.max_length//2])
        encoded = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )

        return {
            "label": torch.tensor(label_id, dtype=torch.long),
            "input_ids": encoded['input_ids'].squeeze(0),
            "attention_mask": encoded['attention_mask'].squeeze(0)
        }
