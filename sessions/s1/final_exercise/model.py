from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """Forward pass through the network, returns the output logits."""
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
            x = self.dropout(x)
        x = self.output(x)

        return nn.functional.log_softmax(x, dim=1)