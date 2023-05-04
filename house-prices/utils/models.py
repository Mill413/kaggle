import time

import torch.nn as nn
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, in_features=331):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        for param in self.model.parameters():
            nn.init.normal_(param, 0, 0.01)

    def forward(self, input):
        return self.model(input)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)


def train(model, optimizer, loss_fn,
          epoch_num, data_loader: DataLoader, step=100):

    start = time.time()
    for epoch in range(epoch_num):

        for x, y in data_loader:
            y_hat = model(x)

            optimizer.zero_grad()
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

        interval = time.time() - start

        if epoch != 0 and epoch % step == 0:
            print(f'epoch-{epoch} time:{interval:.2f}s')
