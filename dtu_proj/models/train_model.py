import torch
import pandas as pd


from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
from dtu_proj.models.model import BERTClassifier
from dtu_proj.data.dataset import UserDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import wandb
import time
from omegaconf import OmegaConf
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device=}')


# previous learning rate was 0.01
def train_epoch(model, dataset, optimizer, loss, epoch, train=True, lr=1e-5, batch_size=1):

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    prefix = 'train' if train else 'val'
    model.train() if train else model.eval()

    with tqdm(DataLoader(dataset, batch_size=batch_size)) as pbar:
        for idx, batch in enumerate(pbar):
            batch = {k: batch[k].to(device) for k in batch.keys()}
            logits = model(batch)
            loss_value = loss(logits, batch['rating'])
            if train:
                loss_value.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            pbar.set_description(f"n: {logits.shape[-1]}  loss: {loss_value.item():.4f}")
            wandb.log(
                {
                    "epoch": epoch,
                    f"{prefix}_loss": loss_value.item(),
                    "lr": lr,
                    "n": logits.shape[-1],
                }
            )


def main(args):

    config = OmegaConf.load(os.path.join('dtu_proj', 'config.yaml'))

    this_time = time.strftime("%Y-%m-%d_%H-%M")

    bert_tokenizer = AutoTokenizer.from_pretrained(config.hp.bert_checkpoint)
    print('tokenizer created')
    dataset = UserDataset(config.hp.data_dir, tokenizer=bert_tokenizer, max_len=512)
    print(f'{len(dataset)=}')

    split_lengths = [int(config.hp.train_ratio * len(dataset)),
                     int(config.hp.val_ratio * len(dataset)),
                     len(dataset) - int(config.hp.train_ratio * len(dataset)) - int(config.hp.val_ratio * len(dataset))]
    print(f'{split_lengths=}')
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=split_lengths,)

    # print(f'{len(train_dataset)=} {len(val_dataset)=} {len(test_dataset)=}')

    model = BERTClassifier(device=device, checkpoint=config.hp.bert_checkpoint, freeze_bert=config.hp.freeze_bert)
    print('BERTClassifier created')
    model.to(device)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.hp.lr)

    wandb_mode = 'online' if args.wandb else 'disabled'
    wandb.init(
        mode=wandb_mode,
        project="mlops",
        save_code=True,
        # track hyperparameters and run metadata
        config={
            "exp_name": f"bert_{this_time}",
            "batch_size": config.hp.batch_size,
            "learning_rate": str(config.hp.lr),
            "epochs": config.hp.epochs,
        },
    )

    for epoch in range(config.hp.epochs):
        print(f'{epoch=}')
        train_epoch(
            model=model,
            dataset=train_dataset,
            epoch=epoch,
            optimizer=optimizer,
            loss=loss,
            train=True,
            lr=config.hp.lr,
            batch_size=config.hp.batch_size)

        with torch.no_grad():
            train_epoch(
                model=model,
                dataset=val_dataset,
                epoch=epoch,
                optimizer=optimizer,
                loss=loss,
                train=False,
                lr=config.hp.lr,
                batch_size=config.hp.batch_size)

        state_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state_dict, f'{config.hp.checkpoint_dir}/bert_e_{str(epoch)}_{this_time}.pt')

    with torch.no_grad():
        train_epoch(
            model=model,
            dataset=test_dataset,
            epoch=0,
            optimizer=optimizer,
            loss=loss,
            train=False,
            lr=config.hp.lr,
            batch_size=config.hp.batch_size)

    loaded_state_dict = torch.load(f'{config.hp.checkpoint_dir}/bert_e_{str(config.hp.epochs-1)}_{this_time}.pt')
    model.load_state_dict(loaded_state_dict['state_dict'])
    model.eval()
    print('model loaded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    print(args)
    main(args)


# # load the data
# data = pd.read_csv('data/processed/clean.csv')
# # split the data
# train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False, random_state=42)
#
# n_factor=300
#
# # get amount of unique books and users
# n_books = data['Book_Id'].nunique()*3
# n_users = data['ID'].nunique()*3
#
# # load previously calculated embeddings
# model = RecommenderNet(n_users, n_books, n_factors=n_factor)
# #model.load_state_dict(torch.load("models/embeddings.pt"))
#
# # train the model
# trained_model = train(model, train_data, test_data)
#
# torch.save(trained_model.state_dict(), "models/model.pt")
