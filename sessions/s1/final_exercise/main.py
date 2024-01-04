import click
import torch
from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel(input_size=784, output_size=10, hidden_layers=[128, 64])
    train_set, _ = mnist()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 30
    print_every = 40
    steps = 0
    running_loss = 0
    running_acc = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in train_loader:
            steps += 1

            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            acc = (output.argmax(dim=1) == labels).float().mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item()

        print(f"Epoch {e+1}/{epochs}..."
              f"Train loss: {running_loss/steps:.3f}..."
              f"Train accuracy: {running_acc/steps:.3f}...")
        running_loss = 0

    print("Saving model")
    torch.save(model.state_dict(), "model.pth")

    # produce a plot of the training loss and accuracy
    # plt.plot(train_losses, label='Training loss')
    # plt.legend(frameon=False)
    # plt.savefig('loss_plot.png')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel(input_size=784, output_size=10, hidden_layers=[128, 64])
    model.load_state_dict(state_dict)

    _, test_set = mnist()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    # accuracy calculation:

    criterion = torch.nn.CrossEntropyLoss()

    print_every = 40
    steps = 0
    running_loss = 0
    running_acc = 0

    model.eval()
    for images, labels in test_loader:
        steps += 1

        # Flatten images into a 784 long vector
        images.resize_(images.size()[0], 784)

        output = model.forward(images)
        loss = criterion(output, labels)
        acc = (output.argmax(dim=1) == labels).float().mean()

        running_loss += loss.item()
        running_acc += acc.item()


    mean_loss = running_loss / steps
    print(f"Test loss: {mean_loss:.3f}...\nTest accuracy: {running_acc/steps:.3f}...")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
