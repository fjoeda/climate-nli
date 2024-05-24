from src.eval.flow import run_nli_inference
import argparse
import pandas as pd
import datetime

results = []

nli_dataset = {
    "name": "climate-fever",
    "data_dir": "./dataset/climate-fever/fever_test.csv",
    "premise_column": "claim",
    "hypothesis_column": "evidence",
    "label_column": "evidence_label",
    "label_encode_dict": {
        "NOT_ENOUGH_INFO": None,
        "SUPPORTS": 0,
        "REFUTES": 1,
    }
}

result = run_nli_inference(
    nli_dataset['data_dir'],
    nli_dataset['premise_column'],
    nli_dataset['hypothesis_column'],
    nli_dataset['label_column'],
    nli_dataset['label_encode_dict'],
)
results.append(result)

df_result = pd.DataFrame(results)
df_result.to_csv(f"./logs/experiment_logs_nli_{datetime.datetime.now().strftime('%Y-%m-%d %H.%M')}.csv")

# term 1 : plan b random (reversed 0 and 1)
# term 2 : plan b no random
# term 3 : plan c random