import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, num_classes, input_size=768, output_features=128):
        super(MLPModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256,bias=True),
            nn.ReLU(),
            nn.Linear(256, output_features,bias=True),
            nn.ReLU(),
            nn.Linear(output_features, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

num_classes = 16
language_model = MLPModel(num_classes)