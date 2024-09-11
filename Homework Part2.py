import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# TensorFlow and tf.keras
import tensorflow as tf
#2.1
train_data = pd.read_csv('kmnist_train.csv')    #这个是panda的dataframe，我直接是把文件放在本地跑的，和ed上面的目录不一样
#上传的时候我会修改相对路径

images = train_data.iloc[:, :-1].values     #提取图像数据和标签
labels = train_data.iloc[:, -1].values      #其实这里不确定，因为他说的是用_kmnist训练数据，但是又说
                                            #utilize kmnist_test.csv in question 2.3.4 only  所以我暂时当作他是对的

image_0 = images[labels == 0][0].reshape(28, 28)    # 这里是选择第一个0和第一个1的样本进行可视化
image_1 = images[labels == 1][0].reshape(28, 28)    #我测试过，会出现模糊的0和1

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.title("Handwritten 0")
plt.imshow(image_0, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Handwritten 1")
plt.imshow(image_1, cmap='gray')
plt.show()


#2.2

X1_train, X1_val, Y1_train, Y1_val = train_test_split(images, labels, test_size=0.3, random_state=42)

#这里就是定义的模型
model_overfit = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])    #跑了一下，这里很早就过拟合了，差不多100个epoch就过拟合了

model_overfit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_overfit = model_overfit.fit(X1_train, Y1_train, epochs=2000, batch_size=128, validation_data=(X1_val, Y1_val))


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=50,restore_best_weights=True, verbose=1)    #在这里我用的早停
model = tf.keras.Sequential([    #我这里使用了早停、Dropout和L2正则化
#实际上这个模型在
    tf.keras.layers.Dense(100, activation='relu', input_shape=(784,), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X1_train,Y1_train,epochs=2000, batch_size=128, validation_data=(X1_val, Y1_val),callbacks=[callback])

#2.3.1


model_overfit.summary()
model.summary()

#2.3.2
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print("Difference between training and validation accuracy:", final_train_acc - final_val_acc)
print("Difference between training and validation loss:", final_train_loss - final_val_loss)


plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(history_overfit.history['accuracy'], label='Training Accuracy')
plt.plot(history_overfit.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Overfitting Example: Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Regularized Model: Training vs Validation Accuracy')
plt.show()

#这个模型大体上是完全可以工作的，但是图像中间的波动非常奇怪（会有一点振荡）

#2.3.4
# 这里用测试集测试
test_data = pd.read_csv('kmnist_test.csv')
x_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)

