from .nli_clf import BertNliClf
from .climabert_clf import ClimateNliClf
from ..utils.eval import generate_report
import pandas as pd
from ..trainer.dataset import prepare_nli_dataset, NliDataset
from ..trainer.flow_nli import eval_model
from ..trainer.model import ClimateNliModel
from torch.utils import data
from transformers import AutoTokenizer
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def run_auto_label(
    autolabel_type,
    df_dir,
    text_column,
    label_column,
    dataset_name,
    export_auto_label=False,
    templates=None
):
    autolabel_type_list = [
        "bart-mnli",
        "climate-nli-binary",
        "climate-nli-unseen-binary"
    ]

    assert autolabel_type in autolabel_type_list

    df_dataset = pd.read_csv(df_dir)
    if dataset_name == "scidcc":
        df_dataset[text_column] = df_dataset['Title'] + " " + df_dataset['Summary'] + " " + df_dataset['Body']
    
    labels = [item.lower() for item in df_dataset[label_column].unique()]
    text_list = df_dataset[text_column].values
    
    if autolabel_type == "bart-mnli":
        model_path = "facebook/bart-large-mnli"
        model = BertNliClf(model_path=model_path, labels=labels, templates=templates)
    elif autolabel_type == "climate-nli-binary":
        model_path = "climatebert/distilroberta-base-climate-f"

        # the weight_path can be changed to suitable value
        weight_path = "./climatebert_binary_best.pth"
        if templates is not None:
            model = ClimateNliClf(
                model_path=model_path,
                weight_path=weight_path,
                labels=labels,
                templates=templates
            )
        else:
            model = ClimateNliClf(
                model_path=model_path,
                weight_path=weight_path,
                labels=labels
            )
    elif autolabel_type == "climate-nli-bench-binary":
        model_path = "climatebert/distilroberta-base-climate-f"

        # the weight_path can be changed to suitable value
        weight_path = "./climatebert_binary_unseen_best.pth"
        if templates is not None:
            model = ClimateNliClf(
                model_path=model_path,
                weight_path=weight_path,
                labels=labels,
                templates=templates
            )
        else:
            model = ClimateNliClf(
                model_path=model_path,
                weight_path=weight_path,
                labels=labels
            )
    
    autolabel = model.run_auto_label(text_list)

    df_dataset[f"{label_column}_auto"] = autolabel

    result = generate_report(
        y_true=[item.lower() for item in df_dataset[label_column].values],
        y_pred=autolabel
    )

    if export_auto_label:
        df_dataset.to_csv(f"{df_dir.replace('.csv', '')}_{dataset_name}_auto_label_{autolabel_type}.csv", index=False)

    return result


def run_nli_inference(
    df_dir,
    premise_column,
    hypothesis_column,
    label_column,
    label_encode_dict,
):
    df = pd.read_csv(df_dir)
    
    df_nli = prepare_nli_dataset(
        df,
        premise_column,
        hypothesis_column,
        label_column,
        label_encode_dict
    )
    
    df_nli.dropna(inplace=True)

    model_path = "climatebert/distilroberta-base-climate-f"

    # the weight_path can be changed to suitable value
    weight_path = "./climatebert_binary_best.pth"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ClimateNliModel(model_path)
    model.to(device)
    best_checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(best_checkpoint)
    test_dataset = NliDataset(df_nli, tokenizer, 512)
    test_dataloader = data.DataLoader(test_dataset, batch_size=8)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    true_list, pred_list, val_loss = eval_model(test_dataloader, model, loss_fn)
    test_report = generate_report(true_list, pred_list, set="test")

    return test_report