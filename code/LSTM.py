import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.conv_1 = nn.Conv1d(200, 128, 3)
        self.bnorm_1 = nn.BatchNorm1d(128)
        self.relu_1 = nn.ReLU6()
        self.pool_1 = nn.MaxPool1d(2)
        self.conv_2 = nn.Conv1d(128, 64, 3)
        self.bnorm_2 = nn.BatchNorm1d(64)
        self.relu_2 = nn.ReLU6()
        self.pool_2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, 1, batch_first=True)
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bnorm_1(x)
        x = self.relu_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.bnorm_2(x)
        x = self.relu_2(x)
        x = self.pool_2(x)
        x = x.transpose(2,1)
        x = self.lstm(x)
        return x

class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(1,1,batch_first=True)
    
    def forward(self, x):
        x = self.lstm(x)
        return x

X = torch.randn(64,200,12)
y = torch.randn(64,200,1)
# X = torch.randn(64,200,1)
# y = torch.randn(64,200,1)
print "generate data"
model = ConvLSTM()
# model = SimpleLSTM()
print "build model"

lr = .01
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

loss = 1.5
iter_times = 0
X.cuda()
y.cuda()
print "enable GPU"

print "training"
while loss > 1e-3 and iter_times < 50:
    outp = model(X)
    loss = loss_fn(outp[0], y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    iter_times += 1
    print "{}/50 - loss: {}".format(iter_times, loss.data.cpu().numpy())