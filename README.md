# Climate-NLI
This repo contains the files for training and evaluating Climate-NLI, a model designed for fact-checking and zero-shot classification on climate-related text.

## How to run
To run the experiment, simply use `python` command followed by `run_*.py`, based on the experiment which you want to run along with the arguments.

### Run the Climate-NLI model training
```
python run_nli_train.py --add_label_variation
```
### Run the Climate-NLI model evaluation
The available model name are :
- `bart-large-mnli`
- `climate-nli-binary`
- `climate-nli-unseen-binary`

or you can defined the name by yourself here and set the condition on `./src/eval/flow.py`.
```
python run_nli_train.py --autolabel_type <model_name>
```

## Citation
TBA