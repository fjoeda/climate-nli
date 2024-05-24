from src.trainer.flow_nli import train_nli_flow
from src.trainer.dataset import prepare_nli_dataset, prepare_nli_dataset_from_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import datetime
import argparse
import random

random.seed(42)

TRAIN_DATASET = [
    {
        "name": "climate-stance",
        "type": "classification",
        "train_dir": "./dataset/climate-stance/train-labeled.csv",
        "val_dir": "./dataset/climate-stance/val-labeled.csv",
        "text_column": "text",
        "label_column": "label",
        "template": "The stance of this tweet regarding to climate change is",
        "label_variation": False
    },
    {
        "name": "SciDCC",
        "type": "classification",
        "train_dir": "./dataset/scidcc/train.csv",
        "val_dir": "./dataset/scidcc/val.csv",
        "text_column": "Text",
        "label_column": "Category",
        "template": "The text is about",
        "label_variation": True
    },
    {
        "name": "climate-eng",
        "type": "classification",
        "train_dir": "./dataset/climate-eng/train-labeled.csv",
        "val_dir": "./dataset/climate-eng/val-labeled.csv",
        "text_column": "text",
        "label_column": "label",
        "template": "The text is about",
        "label_variation": True
    },
    {
        "name": "climate-commitment",
        "type": "classification",
        "train_dir": "./dataset/climate-commitment/climate_commit_train.csv",
        "val_dir": None,
        "text_column": "text",
        "label_column": "label",
        "template": "Does text talk about climate commitment action?",
        "label_variation": False
    },
    {
        "name": "climate-environmental-claim",
        "type": "classification",
        "train_dir": "./dataset/climate-environmental-claim/env_claim_train.csv",
        "val_dir": None,
        "text_column": "text",
        "label_column": "label",
        "template": "Does the claim relate to environment?",
        "label_variation": False
    },
    {
        "name": "climate-detection",
        "type": "classification",
        "train_dir": "./dataset/climate-detection/climate_detect_train.csv",
        "val_dir": None,
        "text_column": "text",
        "label_column": "label",
        "template": "Does the text relate to environment?",
        "label_variation": False
    },
    {
        "name": "climate-sentiment",
        "type": "classification",
        "train_dir": "./dataset/climate-sentiment/sent_train.csv",
        "val_dir": None,
        "text_column": "text",
        "label_column": "label",
        "template": "The text sentiment regarding climate change is",
        "label_variation": False
    },
    {
        "name": "climate-specificity",
        "type": "classification",
        "train_dir": "./dataset/climate-specificity/spec_train.csv",
        "val_dir": None,
        "text_column": "text",
        "label_column": "label",
        "template": "Regarding climate change, the text is",
        "label_variation": False
    },
    {
        "name": "tcfd-recommendation",
        "type": "classification",
        "train_dir": "./dataset/tcfd-recommendation/tcfd_train.csv",
        "val_dir": None,
        "text_column": "text",
        "label_column": "label",
        "template": "Regarding climate recommendation, the text is about",
        "label_variation": False
    },
    {
        "name": "climate-fever",
        "type": "fact-check",
        "train_dir": "./dataset/climate-fever/fever_train.csv",
        "val_dir": None,
        "premise_column": "claim",
        "hypothesis_column": "evidence",
        "label_column": "evidence_label",
        "label_encode_dict": {
            "NOT_ENOUGH_INFO": None,
            "SUPPORTS": 0,
            "REFUTES": 1,
        }
    },
]


parser = argparse.ArgumentParser(
    prog="Autolabeling Experiment"
)

parser.add_argument('--test', action="store_true")
parser.add_argument('--add_label_variation', action="store_true")

args = parser.parse_args()

if args.test:
    print("running experiment in test mode")
    config = json.load(open("./config/nli_train_test_config.json"))
    MODELS = [
        {
            "name": "climatebert",
            "model_path": "climatebert/distilroberta-base-climate-f",
        }
    ]
else:
    print("running experiment in HPC")
    config = json.load(open("./config/nli_train_hpc_config.json"))
    MODELS = [
        {
            "name": "climatebert",
            "model_path": "climatebert/distilroberta-base-climate-f",
        },
    ]

print(config)

df_all_train = pd.DataFrame()
df_all_val = pd.DataFrame()

print("preparing dataset")
for dataset in TRAIN_DATASET:
    print(f"adding {dataset['name']}")
    df_train = pd.read_csv(dataset['train_dir'])
    if dataset['val_dir'] is None:
        df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42)
    else:
        df_val = pd.read_csv(dataset['val_dir'])

    if dataset['type'] == "classification":
        label_variation = dataset['label_variation'] and args.add_label_variation
        if dataset['name'] == "SciDCC":
            df_train[dataset['text_column']] = df_train['Title'] + " " + df_train['Summary'] + " " + df_train['Body']
            df_val[dataset['text_column']] = df_val['Title'] + " " + df_val['Summary'] + " " + df_val['Body']

        df_train = prepare_nli_dataset_from_classification(
            df_clf=df_train,
            text_column=dataset['text_column'],
            label_column=dataset['label_column'],
            label_template=dataset['template'],
            add_label_variation=label_variation,
            random_contradict=True
        )
        
        df_train.dropna(inplace=True)

        df_val = prepare_nli_dataset_from_classification(
            df_clf=df_val,
            text_column=dataset['text_column'],
            label_column=dataset['label_column'],
            label_template=dataset['template'],
            add_label_variation=label_variation,
            random_contradict=True
        )
        
        df_train.dropna(inplace=True)
        
        
    else:
        df_train = prepare_nli_dataset(
            df_nli=df_train,
            premise_column=dataset['premise_column'],
            hypothesis_column=dataset['hypothesis_column'],
            label_column=dataset['label_column'],
            label_encode_dict=dataset['label_encode_dict'],
        )
        
        df_train.dropna(inplace=True)

        df_val = prepare_nli_dataset(
            df_nli=df_val,
            premise_column=dataset['premise_column'],
            hypothesis_column=dataset['hypothesis_column'],
            label_column=dataset['label_column'],
            label_encode_dict=dataset['label_encode_dict'],
        )
        
        df_val.dropna(inplace=True)
    
    df_all_train = pd.concat([df_all_train, df_train])
    df_all_val = pd.concat([df_all_val, df_val])

    if args.test:
        break

reports = []

print(f"# training samples: {df_all_train.shape[0]}")
print(f"# validation samples: {df_all_val.shape[0]}")

for model in MODELS:
    print(f"training {model['name']}")
    report = train_nli_flow(
        df_all_train,
        df_all_val,
        config,
        model['model_path'],
        model['name'] + "_binary"
    )

    reports.append(report)

df_report = pd.DataFrame(reports)
df_report.to_csv(f"./logs/nli_train_bin_{datetime.datetime.now().strftime('%Y-%m-%d %H.%M')}.csv", index=False)
