import torch
from transformers import AutoModel, AutoTokenizer
from torch import nn
from torch.utils import data
from tqdm import tqdm
import numpy as np
from nltk import wordpunct_tokenize

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NliDataset(data.Dataset):
    def __init__(self, df_nli, tokenizer):
        self.df_nli = df_nli
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df_nli)

    def __getitem__(self, idx):
        # label_id, evidence, claim = self.df_nli.values[idx]
        label_id, premise, hypothesis = self.df_nli.values[idx]
        premise = " ".join(self.tokenizer.tokenize(premise)[:self.max_length//2])
        hypothesis = " ".join(self.tokenizer.tokenize(hypothesis)[:self.max_length//2])
        encoded = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )

        return {
            "label": torch.tensor(label_id, dtype=torch.long),
            "input_ids": encoded['input_ids'].squeeze(0),
            "attention_mask": encoded['attention_mask'].squeeze(0)
        } 


class ClimateNliModel(nn.Module):
    def __init__(self, model_path, num_output) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, num_output)

    def forward(self, x, attn):
        x = self.bert(x, attention_mask=attn)
        pooled = self.drop(x.pooler_output)
        out = self.out(pooled)
        return out


class ClimateNliClf:
    def __init__(self, model_path, weight_path, labels, templates="This example is about") -> None:
        self.model = ClimateNliModel(model_path, 2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.labels = labels
        self.templates=templates
        self.model.to(device)
        model_state = torch.load(weight_path, map_location=device)
        self.model.load_state_dict(model_state)

    def decide_label(self, text):
        entailment_score = []
        for item in self.labels:
            hypothesis = f"{self.templates} {item}"
            text = " ".join(wordpunct_tokenize(text)[:512//2])
            hypothesis = " ".join(wordpunct_tokenize(hypothesis)[:512//2])
            encoded = self.tokenizer.encode_plus(
                text,
                hypothesis,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
                truncation=True
            )
            with torch.no_grad():
                out = self.model(
                    encoded['input_ids'].to(device),
                    encoded['attention_mask'].to(device)
                )
            logit = torch.softmax(out[:, [0, 1]], dim=1)[0]
            out = logit[0].item()
            entailment_score.append(out)

        label_idx = np.argmax(entailment_score)
        return self.labels[label_idx]

    def run_auto_label(self, text_list):
        labels = []
        for text in tqdm(text_list):
            label = self.decide_label(text)
            labels.append(label)

        return labels

    

