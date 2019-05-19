import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.conv_1 = nn.Conv1d(12, 32, 3, padding=1)
        self.bnorm_1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU6()
        self.pool_1 = nn.MaxPool1d(2)
        self.conv_2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bnorm_2 = nn.BatchNorm1d(64)
        self.pool_2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, 32, batch_first=True)
        self.dropout = nn.Dropout(.2)
        self.fc_1 = nn.Linear(32, 16)
        self.sigmoid = nn.Sigmoid()
        self.outp = nn.Linear(16, 1)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.outp.weight)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bnorm_1(x)
        x = self.relu(x)
        # x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.bnorm_2(x)
        x = self.relu(x)
        # x = self.pool_2(x)
        x = x.transpose(2,1)
        x = self.lstm(x)
        x = self.dropout(x[0])
        x = self.fc_1(x)
        x = self.sigmoid(x)
        o = self.outp(x)
        return o

class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(1,1,batch_first=True)

    def forward(self, x):
        x = self.lstm(x)
        return x

# X = torch.randn(64,200,12)
# y = torch.randn(64,200,1)
# X = torch.randn(64,200,1)
# y = torch.randn(64,200,1)

npzfile = np.load("earthquake_train.npz")
X_train = npzfile['X_train'].reshape((-1,1201,12)).transpose(0,2,1)
y_train = npzfile['y_train'].reshape((-1,1201,1))
n_samples = X_train.shape[0]
# print X_train.shape, y_train.shape

# raw_data = pd.read_csv("../train.csv", dtype={"acoustic_data":np.int64, "time_to_failure":np.float64}).data
# X_train = raw_data[:4000*150000,0].reshape((4000,150000,1))
# y_train = raw_data[:4000*150000,1].reshape((4000,150000,1))
# X_test = raw_data[4000*150000+1:,0]
# y_test = raw_data[4000*150000+1:,1]

print ("generate data - size: {}".format(n_samples))

model = ConvLSTM().cuda()
# model = SimpleLSTM()
print ("build model")

lr = .0001
batch_size = 64
epoches = 200
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 120, gamma=.1)

epoch = 0
batch_iter = 0
batchs = int(math.ceil(1.*n_samples/batch_size))
rand_index = np.arange(n_samples)
X = torch.from_numpy(X_train).type(torch.FloatTensor).cuda()
y = torch.from_numpy(y_train).type(torch.FloatTensor).cuda()
print ("enable GPU")

summary_writer = SummaryWriter()

print ("training")
while epoch < epoches:
    avg_loss = 0
    np.random.shuffle(rand_index)
    for i in range(batchs):
        batch_iter += 1
        start = i*batch_size
        end = min(start+batch_size, n_samples)
        batch_index = rand_index[start:end]
        outp = model(X[batch_index])
        loss = loss_fn(outp, y[batch_index])
        avg_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        summary_writer.add_scalar("batch loss", loss, global_step=batch_iter)
    epoch += 1
    avg_loss /= batchs
    summary_writer.add_scalar("epoch loss", avg_loss, global_step=epoch)
    print ("{}/{} - avg loss: {} - lr: {}".format(epoch, epoches, avg_loss.cpu().data.numpy(), scheduler.get_lr()[0]))
    scheduler.step()

# summary_writer.add_graph(model)
torch.save(model.state_dict(), "pytorch_model.pt")
summary_writer.close()
print ("Model saved")
