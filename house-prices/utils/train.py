import torch
import torch.nn as nn
import torch.utils.data as data
from torch import Tensor


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    loss = nn.MSELoss()
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


def run(net, loss, optimizer,
        epochs, batch_size,
        train_features, train_labels,
        valid_features, valid_labels=None,
        device='cpu'):
    net.to(device)

    train_ls, valid_ls = [], []

    train_dataset = data.TensorDataset(train_features, train_labels)
    train_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for x, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(x), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
    if valid_labels is not None:
        valid_ls.append(log_rmse(net, valid_features, valid_labels))

    return train_ls, valid_ls


def get_net(in_features=331, out_features=1):
    return nn.Sequential(nn.Linear(in_features, out_features))


def get_k_fold_data(k: int, i: int, X: Tensor, y: Tensor):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train: Tensor | None = None
    y_train: Tensor | None = None
    X_valid: Tensor | None = None
    y_valid: Tensor | None = None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part = X[idx, :]
        y_part = y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, batch_size, loss, lr, wd):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        fold_data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        train_ls, valid_ls = run(net, loss, opt,
                                 num_epochs, batch_size,
                                 *fold_data)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


def valid(train_features, train_labels, k,  epochs, batch_size, lr, weight_decay):
    # k折验证寻找最优超参数
    train_l, valid_l = k_fold(k, train_features, train_labels, epochs, batch_size, nn.MSELoss(), lr, weight_decay)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')
