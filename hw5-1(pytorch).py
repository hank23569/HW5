import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
import os
import datetime

# 檢查 scikit-learn 版本並進行相應的修正
from sklearn import __version__ as sklearn_version
if sklearn_version >= "1.2.0":
    sparse_output_param = {'sparse_output': False}
else:
    sparse_output_param = {'sparse': False}

# 載入資料集
iris = load_iris()
X, y = iris.data, iris.target

# 使用 OneHotEncoder 編碼目標變數
y = OneHotEncoder(**sparse_output_param).fit_transform(y.reshape(-1, 1))

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# 建立資料集
train_dataset = TensorDataset(X_train, torch.max(y_train, 1)[1])
test_dataset = TensorDataset(X_test, torch.max(y_test, 1)[1])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 建立模型
class IrisModel(pl.LightningModule):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # 計算準確率
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # 記錄訓練損失和準確率
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # 計算準確率
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # 記錄驗證損失和準確率
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# 設定 TensorBoard 日誌目錄
log_base_dir = "C:/logs/fit_lightning"  # 確保目錄是全英文
log_dir = os.path.join(log_base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)

# 設定 TensorBoard logger
logger = pl.loggers.TensorBoardLogger(save_dir=log_base_dir, name="iris_logs")

# 訓練模型
trainer = pl.Trainer(max_epochs=50, log_every_n_steps=1, logger=logger)
model = IrisModel()
trainer.fit(model, train_loader, test_loader)

# 提示啟動 TensorBoard
print(f"啟動 TensorBoard 使用以下命令: tensorboard --logdir {log_base_dir}")
