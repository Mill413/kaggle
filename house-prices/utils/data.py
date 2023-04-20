import os

import pandas as pd
import torch


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
