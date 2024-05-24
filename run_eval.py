from src.eval.flow import run_auto_label
import argparse
import pandas as pd
import datetime

parser = argparse.ArgumentParser(
    prog="Autolabeling Experiment"
)

parser.add_argument("--autolabel_type", required=True)

args = parser.parse_args()

results = []

dataset_list = [
    {   
        "dataset_name": "climate_eng",
        "dataset_dir": "./dataset/climate-eng/test-labeled.csv",
        "text_column": "text",
        "label_column": "label",
        "template": "This example is about"
    },
    {   
        "dataset_name": "climate_stance",
        "dataset_dir": "./dataset/climate-stance/test-labeled.csv",
        "text_column": "text",
        "label_column": "label",
        "template": "The stance of this tweet regarding to climate change is"
    },
    {   
        "dataset_name": "scidcc",
        "dataset_dir": "./dataset/scidcc/test.csv",
        "text_column": "Text",
        "label_column": "Category",
        "template": "This example is about"
    },
    {
        "dataset_name": "climate-commitment",
        "dataset_dir": "./dataset/climate-commitment/climate_commit_test.csv",
        "text_column": "text",
        "label_column": "label",
        "template": "Does text talk about climate commitment action?",
    },
    {
        "dataset_name": "climate-environmental-claim",
        "dataset_dir": "./dataset/climate-environmental-claim/env_claim_test.csv",
        "text_column": "text",
        "label_column": "label",
        "template": "Does the claim relate to environment?",
    },
    {
        "dataset_name": "climate-sentiment",
        "dataset_dir": "./dataset/climate-sentiment/sent_test.csv",
        "text_column": "text",
        "label_column": "label",
        "template": "The text sentiment regarding climate change is",
    },
    {
        "dataset_name": "climate-specificity",
        "dataset_dir": "./dataset/climate-specificity/spec_test.csv",
        "text_column": "text",
        "label_column": "label",
        "template": "The text is climate change",
    },
    {
        "dataset_name": "tcfd-recommendation",
        "dataset_dir": "./dataset/tcfd-recommendation/tcfd_test.csv",
        "text_column": "text",
        "label_column": "label",
        "template": "Regarding climate recommendation, the text is about",
    },
    {
        "dataset_name": "climate-detection",
        "dataset_dir": "./dataset/climate-detection/climate_detect_test.csv",
        "text_column": "text",
        "label_column": "label",
        "template": "Does the text related to climate?",
    },
]

for dataset in dataset_list:
    print(f"Autolabel dataset : {dataset['dataset_name']}")
    result = run_auto_label(
        autolabel_type=args.autolabel_type,
        df_dir=dataset['dataset_dir'],
        text_column=dataset['text_column'],
        label_column=dataset['label_column'],
        dataset_name=dataset['dataset_name'],
        export_auto_label=True,
        templates=dataset['template']
    )

    result["dataset_name"] = dataset["dataset_name"]

    results.append(result)

df_result = pd.DataFrame(results)
df_result.to_csv(f"./logs/experiment_logs_{args.autolabel_type}_{datetime.datetime.now().strftime('%Y-%m-%d %H.%M')}.csv")