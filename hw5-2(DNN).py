import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import datetime
import os

# 函數: 繪製訓練和驗證曲線
def plot_training_history(history, model_type):
    plt.figure(figsize=(12, 5))
    # 損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 準確率曲線
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_type} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 確保日誌路徑的穩定性
def ensure_log_dir_exists(base_dir, sub_dir):
    log_dir = os.path.join(base_dir, sub_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# 載入 MNIST 資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 資料預處理
X_train_flat = X_train.reshape((X_train.shape[0], 28 * 28)).astype("float32") / 255
X_test_flat = X_test.reshape((X_test.shape[0], 28 * 28)).astype("float32") / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 建立 DNN 模型
dnn_model = models.Sequential([
    layers.Input(shape=(28 * 28,)),  # 使用 Input 避免警告
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 設定 TensorBoard 日誌
log_dir_dnn = ensure_log_dir_exists("C:/logs", "mnist_dnn")
tensorboard_callback_dnn = tf.keras.callbacks.TensorBoard(log_dir=log_dir_dnn, histogram_freq=1)

# 訓練 DNN 模型
print("Training DNN Model...")
dnn_history = dnn_model.fit(X_train_flat, y_train, epochs=10, batch_size=128,
                            validation_data=(X_test_flat, y_test), callbacks=[tensorboard_callback_dnn])

# 繪製 DNN 訓練歷史
plot_training_history(dnn_history, "DNN")

# 評估模型
dnn_test_loss, dnn_test_acc = dnn_model.evaluate(X_test_flat, y_test)
print(f"DNN Test Accuracy: {dnn_test_acc:.4f}")

# 提示啟動 TensorBoard
print(f"啟動 TensorBoard 使用以下命令: tensorboard --logdir C:/logs/mnist_dnn")
