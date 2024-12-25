import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import datetime

# 載入資料集
iris = load_iris()
X, y = iris.data, iris.target

# 使用 OneHotEncoder 編碼目標變數
y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 設定日誌目錄到一個完全英文的路徑
log_base_dir = "C:/logs/fit"  # 確保路徑中不含中文
log_dir = os.path.join(log_base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)  # 確保目錄存在

# 建立模型
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),  # 定義輸入形狀
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 設定 TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 訓練模型
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# 打印模型摘要
model.summary()

# 提示啟動 TensorBoard
print(f"啟動 TensorBoard 使用以下命令: tensorboard --logdir {log_base_dir}")
