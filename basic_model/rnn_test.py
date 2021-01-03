import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.full_line = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        # init state
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)

        # forward
        out, _ = self.lstm(x, (h0, c0))

        # finally decode the hidden state of the last/first time step
        h_l = out[:, -1, :]
        out = self.full_line(h_l)
        return out

model = BiRNN(input_size=28, hidden_size=128, num_layers=2, num_classes=10)
criterion = nn.CrossEntropyLoss()   #使用交叉熵作为损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)	# 优化器，用于在反向传播中进行参数更新

input_data = torch.randn(100,28,28)
input_data = input_data.reshape(-1, 28, 28)


outputs = model(input_data)
loss = criterion(outputs, labels)

