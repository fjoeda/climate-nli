from transformers import AutoModel, RobertaModel
from torch import nn

# initiate BERT-based model object for NLI tasks
class ClimateNliModel(nn.Module):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, x, attn):
        x = self.bert(x, attention_mask=attn)
        pooled = self.drop(x.pooler_output)
        out = self.out(pooled)
        return out
