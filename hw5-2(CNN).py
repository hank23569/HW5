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
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype("float32") / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype("float32") / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 建立 CNN 模型
cnn_model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 設定 TensorBoard 日誌
log_dir_cnn = ensure_log_dir_exists("C:/logs", "mnist_cnn")
tensorboard_callback_cnn = tf.keras.callbacks.TensorBoard(log_dir=log_dir_cnn, histogram_freq=1)

# 訓練 CNN 模型
print("Training CNN Model...")
cnn_history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=128,
                            validation_data=(X_test, y_test), callbacks=[tensorboard_callback_cnn])

# 繪製 CNN 訓練歷史
plot_training_history(cnn_history, "CNN")

# 評估模型
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test, y_test)
print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")

# 提示啟動 TensorBoard
print(f"啟動 TensorBoard 使用以下命令: tensorboard --logdir C:/logs/mnist_cnn")
