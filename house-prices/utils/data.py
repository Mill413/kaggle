import os

import pandas as pd
import torch
import torch.utils.data as data
from torch import nn, Tensor


def get_data(train_path="data/train.csv", test_path="data/test.csv"):
    cwd = os.getcwd()
    train_csv_path = os.path.join(cwd, train_path)
    test_csv_path = os.path.join(cwd, test_path)

    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)

    return train_data, test_data


def process(train_data, test_data):
    # 删除ID标签并连接训练测试数据集，以统一处理
    all_data = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))

    # 筛选出所有数字类型数据，并将其标准化
    numeric_features = all_data.dtypes[all_data.dtypes != 'object'].index
    all_data[numeric_features] = all_data[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )

    # 将缺失值设为0
    all_data[numeric_features] = all_data[numeric_features].fillna(0)

    # 处理离散值
    all_data = pd.get_dummies(all_data, dummy_na=True)

    # 提取数据并转换为张量
    train_cnt = train_data.shape[0]
    train_features = torch.tensor(all_data[:train_cnt].values, dtype=torch.float32)
    test_features = torch.tensor(all_data[train_cnt:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    return train_features, test_features, train_labels


def load_data(features, labels, batch_size):
    ds = data.TensorDataset(features, labels)
    return data.DataLoader(ds, batch_size)


def log_rmse(net, loss, x, y):
    with torch.no_grad:
        y_hat = torch.log(torch.clamp(net(x), 1, float('inf')))
        res = torch.sqrt(loss(y_hat, torch.log(y)))
    return res.item()


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
