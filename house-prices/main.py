import pandas as pd
import torch
import torch.nn as nn

import utils.data as ud
import utils.models as um

train_data, test_data = ud.get_data()
train_features, test_features, train_labels = ud.process(train_data, test_data)

# 超参数
k, epochs, lr, weight_decay, batch_size = 8, 500, 0.1, 300, 64

# 基础设施
net = um.MLP()
opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
loss = nn.MSELoss()

um.train(net, opt, loss, epochs,
         ud.load_data(train_features, train_labels, batch_size))


print(ud.log_rmse(net, loss, train_features, train_labels))


# 预测
preds = net(test_features).detach().numpy()
test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)
