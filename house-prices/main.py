import pandas as pd
import torch
import torch.nn as nn

import utils.data as ld
import utils.train as lt

train_data, test_data = ld.get_data()
train_features, test_features, train_labels = ld.process(train_data, test_data)

# 超参数
k, epochs, lr, weight_decay, batch_size = 5, 1000, 10, 0, 64

# k折验证
lt.valid(train_features, train_labels, k, epochs, batch_size, lr, weight_decay)

# 基础设施
net = lt.get_net()
opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
loss = nn.MSELoss()

# 训练
train_ls, _ = lt.run(net, loss, opt,
                     epochs, batch_size,
                     train_features, train_labels, None, None)
print(f'训练log rmse：{float(train_ls[-1]):f}')

# 预测
preds = net(test_features).detach().numpy()
test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)
