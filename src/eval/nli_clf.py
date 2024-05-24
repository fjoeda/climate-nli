import torch
from transformers import pipeline
from tqdm import tqdm

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class BertNliClf:
    def __init__(self, model_path, labels, templates) -> None:
        self.model = pipeline(
            "zero-shot-classification",
            model=model_path,
            device=0
        )
        self.labels = labels
        if templates is not None:
            self.templates = templates + " {}."
        else:
            self.templates = "This example is {}."
    
    def decide_label(self, text):
        result = self.model(text, self.labels, hypothesis_template=self.templates, multi_label=False)
        return result['labels'][0]
    
    def run_auto_label(
        self,
        text_list,
    ):
        labels = []
        for text in tqdm(text_list):
            label = self.decide_label(text)
            labels.append(label)

        return labels
