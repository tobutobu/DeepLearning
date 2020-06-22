import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from datetime import datetime

CATEGORIES = ["dogs", "cats"]
NUM_CLASSES = len(CATEGORIES)
IMAGE_SIZE = 32

"""
データの読み込み
"""


def load_data():
    X_train, X_test, y_train, y_test = np.load("./dataset.npy", allow_pickle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.175)
    X_train = X_train.astype("float") / 256
    X_valid = X_valid.astype("float") / 256
    X_test = X_test.astype("float") / 256

    y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
    y_valid = np_utils.to_categorical(y_valid, NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


"""
モデルの生成
"""


def create_model(num_layer, activation, mid_units, num_filters, dropout_rate):
    """
    num_layer : 畳込み層の数
    activation : 活性化関数
    mid_units : FC層のユニット数
    num_filters : 各畳込み層のフィルタ数
    dropout_rate : 入力ユニットをドロップする割合
    """

    model = Sequential()
    model.add(Conv2D(num_filters[0], (3, 3), padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(Activation(activation))

    for i in range(1, num_layer):
        model.add(Conv2D(num_filters[i], (3, 3), padding='same'))
        model.add(Activation(activation))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())

    model.add(Dense(mid_units))
    model.add(Activation(activation))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


"""
パラメータの探索
"""


def objective(trial):
    # セッションのクリア
    K.clear_session()

    # データ読み込み
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

    # 最適化するパラメータの設定
    # 畳込み層の数
    num_layer = trial.suggest_int("num_layer", 3, 8)

    # FC層のユニット数
    mid_units = int(trial.suggest_discrete_uniform("mid_units", 300, 500, 10))

    # 各畳込み層のフィルタ数
    num_filters = [int(trial.suggest_discrete_uniform("num_filter_" + str(i), 32, 512, 32)) for i in range(num_layer)]

    # 活性化関数
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid"])

    # optimizer
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])

    # Dropout
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)

    # 学習率
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e0)
    model = create_model(num_layer, activation, mid_units, num_filters, dropout_rate)

    if optimizer == "sgd":
        opt = keras.optimizers.SGD(learning_rate)
    elif optimizer == "adam":
        opt = keras.optimizers.adam(learning_rate)
    elif optimizer == "rmsprop":
        opt = keras.optimizers.rmsprop(learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=64,
                        epochs=10,
                        validation_data=(X_valid, y_valid))

    return -np.amax(history.history['val_accuracy'])


"""
実行

"""
# 開始時刻
datetime.now().strftime("%Y/%m/%d %H:%M:%S")

study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(study.best_params)
print(study.best_value)
print(study.best_trial)
# 終了時刻
datetime.now().strftime("%Y/%m/%d %H:%M:%S")