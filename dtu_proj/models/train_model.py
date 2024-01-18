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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device=}')


# previous learning rate was 0.01
def train(model, dataset, epoch, prefix='train', lr=0.1, batch_size=32, step_size=2, gamma=0.1):

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    with tqdm(DataLoader(dataset, batch_size=1)) as pbar:
        for idx, batch in enumerate(pbar):
            batch = {k: batch[k].to(device) for k in batch.keys()}
            logits = model(batch)
            loss_value = loss(logits, batch['rating'])
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"n: {logits.shape[-1]}\tloss: {loss_value.item():.4f}")
            pbar.update()
            wandb.log(
                {
                    "epoch": epoch,
                    f"{prefix}_loss": loss_value.item(),
                    "lr": lr,
                    "n": logits.shape[-1],
                }
            )

    # for epoch in range(epochs):
    #     progress_bar = tqdm(train_dataloader, desc='Epoch {:03d}'.format(epoch + 1), leave=False, disable=False)
    #
    #     for user, book, rating in train_dataloader:
    #
    #         # Forward pass, have to stack them like so to do forward pass
    #         outputs = model(torch.stack((user, book),dim=1))
    #         loss = criterion(outputs, rating.float().unsqueeze(1))
    #
    #         # Backward pass and optimization
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #
    #         progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
    #         progress_bar.update()
    #     # Validation loss
    #     model.eval()
    #     total_test_loss = 0
    #     with torch.no_grad():
    #         for user, book, rating in test_dataloader:
    #             test_outputs = model(torch.stack((user, book), dim=1))
    #             test_loss = criterion(test_outputs, rating.float().unsqueeze(1))
    #             total_test_loss += test_loss.item()
    #     avg_test_loss = total_test_loss / len(test_dataloader)
    #     model.train()
    #
    #     scheduler.step()
    #     print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {avg_test_loss}, Learning Rate: {scheduler.get_last_lr()[0]}')
    #
    # return model


def main(args):

    epochs = 2
    lr = 1e-4
    data_dir = 'data'
    checkpoint = "prajjwal1/bert-tiny"  # L=2, H=128
    BATCH_SIZE = 1
    this_time = time.strftime("%Y-%m-%d_%H-%M")

    bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print('tokenizer created')
    dataset = UserDataset(data_dir, tokenizer=bert_tokenizer, max_len=512)
    print(f'{len(dataset)=}')
    model = BERTClassifier(device=device, checkpoint=checkpoint)
    print('BERTClassifier created')
    model.to(device)

    wandb_mode = 'online' if args.wandb else 'disabled'
    wandb.init(
        mode=wandb_mode,
        project="mlops",
        save_code=True,
        # track hyperparameters and run metadata
        config={
            "exp_name": f"bert_e_{str(epochs)}_b_{str(BATCH_SIZE)}_{this_time}_lr_{str(lr)}",
            "batch_size": BATCH_SIZE,
            "learning_rate": str(lr),
            "epochs": epochs,
        },
    )

    for epoch in range(epochs):
        print(f'{epoch=}')
        train(model, dataset, epoch, prefix='train', lr=lr)


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
