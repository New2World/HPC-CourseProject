import os, math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import mean_absolute_error

class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.conv_1 = nn.Conv1d(36, 64, 3, padding=1)
        self.bnorm_1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU6()
        self.pool_1 = nn.MaxPool1d(2)
        self.conv_2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bnorm_2 = nn.BatchNorm1d(128)
        self.pool_2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.dropout = nn.Dropout(.5)
        self.fc_1 = nn.Linear(64, 16)
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

model = ConvLSTM()

if os.path.exists("../model_param/pytorch_model_enhanced.pt"):
    npzfile = np.load("../data/earthquake_test_enhanced.npz")
    X_test = npzfile['X_test'].reshape((1,-1,36)).transpose(0,2,1)
    y_test = npzfile['y_test'].reshape((1,-1,1))
    n_samples = X_test.shape[0]
    print ("load data")

    model.load_state_dict(torch.load("../model_param/pytorch_model_enhanced.pt"))
    model.eval()
    model.cuda()
    print ("load model")

    print ("evaluating")
    X = torch.from_numpy(X_test).type(torch.FloatTensor).cuda()
    y_pred = model(X).detach().cpu().numpy()

    mae = mean_absolute_error(y_test.squeeze(), y_pred.squeeze())

    print ("MAE: {}".format(mae))
else:
    npzfile = np.load("../data/earthquake_train_enhanced.npz")
    X_train = npzfile['X_train'].reshape((1,8000,36)).transpose(0,2,1)
    y_train = npzfile['y_train'].reshape((1,8000,1))
    n_samples = X_train.shape[0]
    print ("load data")

    model.cuda()
    print ("build model")

    lr = .0045
    batch_size = 64
    epoches = 1000
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 300, gamma=.1)

    epoch = 0
    batch_iter = 0
    batchs = int(math.ceil(1.*n_samples/batch_size))
    seq_index = np.arange(n_samples)
    X = torch.from_numpy(X_train).type(torch.FloatTensor).cuda()
    y = torch.from_numpy(y_train).type(torch.FloatTensor).cuda()
    print ("enable GPU")

    summary_writer = SummaryWriter()

    print ("training")
    while epoch < epoches:
        outp = model(X)
        loss = loss_fn(outp, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch += 1
        summary_writer.add_scalar("epoch loss", loss, global_step=epoch)
        print ("{}/{} - avg loss: {} - lr: {}".format(epoch, epoches, loss.cpu().data.numpy(), scheduler.get_lr()[0]))
        scheduler.step()

    # summary_writer.add_graph(model)
    torch.save(model.state_dict(), "pytorch_model_enhanced.pt")
    summary_writer.close()
    print ("Model saved")
