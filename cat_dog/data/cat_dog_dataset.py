import os, sys
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

import cv2
import random, math
import numpy as np
import pickle

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/data.pkl"

DATADIR = "data\PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 32
training_data = []


def create_training_data():
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name),)  # 画像読み込み
                img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # 画像のリサイズ
                training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加
            except Exception as e:
                pass


# データとラベルを作成
create_training_data()

random.shuffle(training_data)  # データをシャッフル
X_train = []  # 画像データ
y_train = []  # ラベル情報

for feature, label in training_data:
    X_train.append(feature)
    y_train.append(label)

# numpy配列に変換
X_train = np.array(X_train)
y_train = np.array(y_train)

# 学習データと検証データに分ける
th = math.floor(len(training_data) * 0.8)
X_test = X_train[th:]
y_test = y_train[th:]
X_train = X_train[0:th]
y_train = y_train[0:th]

key_file = {
    'train_img': X_train,
    'train_label': y_train,
    'test_img': X_test,
    'test_label': y_test
}

# データセット作成
def _create_dataset():
    dataset = {}
    dataset['train_img'] = key_file['train_img']
    dataset['train_label'] = key_file['train_label']
    dataset['test_img'] = key_file['test_img']
    dataset['test_label'] = key_file['test_label']

    return dataset

# データセット保存
def init_mnist():
    dataset = _create_dataset()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 2))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_data(normalize=True, plot_show=True, one_hot_label=False):
    """MNISTデータセットの読み込み

    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    Plot_show : Plot.show()を使って画像を表示させる

    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    if not os.path.exists(save_file):
            init_mnist()

    with open(save_file, 'rb') as f:
            dataset = pickle.load(f)

    # dataset = _create_dataset()

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not plot_show:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].transpose(0, 3, 2, 1)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

if __name__ == '__main__':
    init_mnist()