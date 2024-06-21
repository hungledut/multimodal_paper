from torchvision.models import resnet18
import torch
import torch.nn as nn
import math

model = resnet18(pretrained=True)

model.fc

class Vision_model(nn.Module):
    def __init__(self, output_features=128):
        super(Vision_model, self).__init__()

        self.resnet18 = resnet18(pretrained=False)

        modelOutputFeats = self.resnet18.fc.in_features

        self.resnet18.fc = nn.Linear(modelOutputFeats,output_features)
    def forward(self, x):

        x = self.resnet18(x)
        return x

class Language_model(nn.Module):
    def __init__(self, input_size=768, output_features=128):
        super(Language_model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256,bias=True),
            nn.ReLU(),
            nn.Linear(256, output_features,bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,
        hidden_size: int,
        num_heads: int,
        bias: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.nh = num_heads
        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        B, S, C = x.shape

        x = self.Wqkv(x).reshape(B, S, 3, self.nh, C//self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        attn = attn.softmax(dim=-1)

        x = attn @ v

        return torch.squeeze(self.Wo(x.transpose(1, 2).reshape(B, S, C)), 1)
    
class MultiCrossAttention(nn.Module):
    def __init__(self,
        hidden_size: int,
        num_heads: int,
        bias: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.nh = num_heads
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.Wkv = nn.Linear(hidden_size, hidden_size * 2, bias=bias)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x, y):
        x = torch.unsqueeze(x, 1)
        B, S, C = x.shape

        q = self.Wq(x).reshape(B, S, self.nh, C//self.nh).transpose(1, 2)
        y = self.Wkv(y).reshape(B, S, 2, self.nh, C//self.nh)
        k, v = y.transpose(3, 1).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        attn = attn.softmax(dim=-1)

        x = attn @ v

        return torch.squeeze(self.Wo(x.transpose(1, 2).reshape(B, S, C)), 1)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(x.shape,self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class ProposedMultimodalModel(nn.Module):
    def __init__(self, Vision_model,Language_model,num_classes,output_features=128):
        super(ProposedMultimodalModel , self).__init__()
        self.vision_model = Vision_model()
        self.language_model = Language_model()

        self.multicrossattention = MultiCrossAttention(128,4)
        self.multihead = MultiHeadAttention(256,4)

        self.norm = nn.LayerNorm(256)
        self.fc = nn.Linear(output_features*2, output_features*2,bias=True)


        self.fc1 = nn.Linear(output_features*2, 128,bias=True)
        self.fc2 = nn.Linear(128, num_classes,bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X1, X2):
        X1 = self.vision_model(X1)
        X2 = self.language_model(X2)

        # Multi head Cross attention
        X1 = self.multicrossattention(X1,X2)
        X2 = self.multicrossattention(X2,X1)

        # Multi head attention
        x = torch.cat([X1,X2],dim=1)
        x_skip = x
        x = self.multihead(x)
        x = torch.add(x,x_skip)
        x = self.norm(x)
        x_skip = x
        x = self.fc(x)
        x = torch.add(x,x_skip)
        x = self.norm(x)

        #Feed Forward
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
