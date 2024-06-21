import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, num_classes, input_size=768, output_features=128):
        super(LSTMModel, self).__init__()
        self.fc = nn.Linear(input_size, 256,bias=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256, output_features,bias=True)
        self.lstm = nn.LSTM(input_size=output_features, hidden_size=num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x,(_1,_2) = self.lstm(x)
        x = self.softmax(x)
        return x

num_classes = 16
language_model = LSTMModel(num_classes)