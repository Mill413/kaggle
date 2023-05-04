import torch
import torch.nn as nn
import torch.utils.data as data
from .models import MLP
from . import data as data


def valid_loss(net, features, labels):
    net.eval()
    return data.log_rmse(net(features), labels)


# def run(net, loss, optimizer,
#         epochs, batch_size,
#         train_features, train_labels,
#         valid_features, valid_labels=None):
#     train_ls, valid_ls = [], []
#
#     train_dataset = data.TensorDataset(train_features, train_labels)
#     train_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     for epoch in range(epochs):
#         for x, y in train_iter:
#             optimizer.zero_grad()
#             l = loss(net(x), y)
#             l.backward()
#             optimizer.step()
#         train_ls.append(valid_loss(net, train_features, train_labels))
#     if valid_labels is not None:
#         valid_ls.append(valid_loss(net, valid_features, valid_labels))
#
#     return train_ls, valid_ls


def k_fold(k, X_train, y_train, num_epochs, batch_size, loss, lr, wd):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        fold_data = data.get_k_fold_data(k, i, X_train, y_train)
        net = MLP()
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        train_ls, valid_ls = run(net, loss, opt,
                                 num_epochs, batch_size,
                                 *fold_data)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


def valid(train_features, train_labels, k, epochs, batch_size, lr, weight_decay):
    # k折验证寻找最优超参数
    train_l, valid_l = k_fold(k, train_features, train_labels, epochs, batch_size, nn.MSELoss(), lr, weight_decay)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')
