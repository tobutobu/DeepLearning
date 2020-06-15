from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np
import matplotlib.pyplot as plt

CATEGORIES = ["dog", "cat"]
NUM_CLASSES = len(CATEGORIES)
IMAGE_SIZE = 64

"""
データを読み込む関数
"""


def load_data():
    X_train, X_test, y_train, y_test = np.load("./dataset.npy", allow_pickle=True)
    # 入力データの各画素値を0-1の範囲で正規化(学習コストを下げるため)
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255
    # to_categorical()にてラベルをone hot vector化
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

    return X_train, y_train, X_test, y_test


"""
モデルを学習する関数
"""


def train(X_train, y_train, X_test, y_test, batch_size, epochs):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # 最適化アルゴリズム
    opt = keras.optimizers.rmsprop(lr=0.00005, decay=1e-6)
    Adam = keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, )
    # モデル設定
    model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
    result = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    # モデルを保存
    model.save('./cat_dog_cnn.h5')

    # 評価 & 評価結果出力
    print(model.evaluate(X_test, y_test))

    # グラフの描画
    x = range(epochs)
    plt.plot(x, result.history['accuracy'], label="training")
    plt.plot(x, result.history['val_accuracy'], label="validation")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return model


"""
データの読み込みとモデルの学習を行う
"""
# データの読み込み
X_train, y_train, X_test, y_test = load_data()
BATCH_SIZE = 28
EPOCHS = 40

# モデルの学習
train(X_train, y_train, X_test, y_test, BATCH_SIZE, EPOCHS)
