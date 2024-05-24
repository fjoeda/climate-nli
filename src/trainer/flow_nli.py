import torch
from torch import nn, optim
from torch.utils import data
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.utils.eval import generate_report
from src.trainer.dataset import NliDataset
from src.trainer.model import ClimateNliModel
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# set random seed for reproducible result
def init_state():
    torch.random.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)

# NLI model training procedure
def train_model(dataloader, model, loss_fn, optimizer, scheduler):
    # set model mode to train
    model.train()
    true_list = torch.tensor([]).to(device)
    pred_list = torch.tensor([]).to(device)
    losses = []

    for data in tqdm(dataloader):
        inputs = data['input_ids'].to(device)
        mask = data['attention_mask'].to(device)
        labels = data['label'].to(device)

        # forward
        output = model(inputs, mask)
        pred = torch.argmax(output, dim=1)

        # add prediction to the list
        true_list = torch.cat([true_list, labels])
        pred_list = torch.cat([pred_list, pred])

        # compute loss
        loss = loss_fn(output, labels)
        losses.append(loss.item())

        # backpropagation, update weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return true_list.tolist(), pred_list.tolist(), torch.mean(torch.tensor(losses)).item()

def eval_model(dataloader, model, loss_fn):
    # set model mode to eval
    model.eval()
    true_list = torch.tensor([]).to(device)
    pred_list = torch.tensor([]).to(device)
    losses = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            labels = data['label'].to(device)

            # forward pass
            output = model(inputs, mask)
            pred = torch.argmax(output, dim=1)

            # add prediction to the list
            true_list = torch.cat([true_list, labels])
            pred_list = torch.cat([pred_list, pred])

            # compute loss
            loss = loss_fn(output, labels)
            losses.append(loss.item())

    return true_list.tolist(), pred_list.tolist(), torch.mean(torch.tensor(losses)).item()

def training_loop(
    train_dataloader,
    val_dataloader,
    model,
    model_name,
    optimizer,
    scheduler,
    loss_fn,
    config,
    test_dataloader=None
):
    # history = {
    #     "train_loss": [],
    #     "train_acc": [],
    #     "val_loss": [],
    #     "val_acc": [],
    # }

    min_loss = 9999
    current_loss = 9999
    patience = 0
    best_acc = 0
    best_epoch = 0
    early_stop_epoch = 0

    # training loop for each epoch
    for i in range(config['num_epoch']):
        print(f"epoch {i+1}")
        true_list, pred_list, train_loss = train_model(train_dataloader, model, loss_fn, optimizer, scheduler)
        train_acc = accuracy_score(true_list, pred_list)
        print(f"train_loss : {train_loss} train_acc: {train_acc}")

        true_list, pred_list, val_loss = eval_model(val_dataloader, model, loss_fn)
        val_acc = accuracy_score(true_list, pred_list)
        print(f"val_loss : {val_loss} val_acc: {val_acc}")

        # check validation loss
        if current_loss > val_loss:
            patience = 0
        else:
            patience += 1

        current_loss = val_loss

        if min_loss >= val_loss:
            # save model checkpoint with the least validation loss
            print(f"save best model on epoch {i+1}")
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            min_loss = val_loss
            
        if best_acc <= val_acc:
            # save model checkpoint with the best accuracy
            print(f"save best acc model on epoch {i+1}")
            best_epoch = i+1
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            best_acc = val_acc

        # implement early stoping
        if patience > config['patience']:
            early_stop_epoch = i+1
            break
    
    # load the best checkpoint
    best_checkpoint = torch.load(f"{model_name}_best.pth", map_location=device)
    model.load_state_dict(best_checkpoint)
    model.eval()

    # evaluate on all dataset
    true_list, pred_list, train_loss = eval_model(train_dataloader, model, loss_fn)
    train_report = generate_report(true_list, pred_list, set="train")
    train_report['train_loss'] = train_loss
    true_list, pred_list, val_loss = eval_model(val_dataloader, model, loss_fn)
    val_report = generate_report(true_list, pred_list, set="val")
    val_report['val_loss'] = val_loss
    if test_dataloader:
        true_list, pred_list, val_loss = eval_model(test_dataloader, model, loss_fn)
        test_report = generate_report(true_list, pred_list, set="test")
    else:
        test_report = {}
    
    print("Best Epoch : ", best_epoch)
    print("Early Stop : ", early_stop_epoch)
    # combine report
    report = {**config, **train_report, **val_report, **test_report}

    return report

def train_nli_flow(
    df_train,
    df_val,
    config,
    model_path,
    model_name,
):
    # initiate random seed
    init_state()

    # initiate tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ClimateNliModel(model_path)
    model.to(device)

    # Initiate dataset and dataloader for training
    train_dataset = NliDataset(df_train, tokenizer, config['max_length'])
    val_dataset = NliDataset(df_val, tokenizer, config['max_length'])
    train_dataloader = data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4)
    val_dataloader = data.DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=4)

    # initiate optimizer, loss function, and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = len(train_dataloader) * config['num_epoch']
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    # perform training loop
    report = training_loop(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        model_name=model_name,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config
    )

    return report