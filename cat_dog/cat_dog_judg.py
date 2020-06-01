import os, sys
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

import cv2
import numpy as np
from deep_conv_net import DeepConvNet
from Rayers.functions import *


# 画像を読み込む
DATADIR = ""
CATEGORIES = ["img"]
IMG_SIZE = 32
training_data = []


# 画像の加工
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

X_data = []    # 画像データ
t_data = []     # ラベル情報

for feature, label in training_data:
    X_data.append(feature)
    t_data.append(label)

# numpy配列に変換
X_data = np.array(X_data)
t_data = np.array(t_data)

X_data = X_data.transpose(0, 3, 2, 1)      # 配列内の位置移動

# ディープな畳み込みニューラルネットワーク生成
network = DeepConvNet(
        input_dim=(3, 32, 32),
        conv_param_1={
            'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_2={
            'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_3={
            'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_4={
            'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_5={
            'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        conv_param_6={
            'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 1
        },
        hidden_size=50, output_size=2
)

# パラメータの読み込み
network.load_params("params_2000.pkl")

# 予測結果
result_data = []
for i in range(t_data.size):
    if softmax(network.predict(X_data, train_flg=False))[i, 0] == 1:
        # print("これは犬ですねぇ！")
        result_data.append('犬')
    else:
        # print("これは猫ですねぇ！")
        result_data.append("猫")
print(result_data)
