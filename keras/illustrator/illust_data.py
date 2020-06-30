import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
from PIL import ImageFile
import random, math
# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

CATEGORIES = ["ponkan", "ixy", "makihituzi", "tokunou", "anmi"]
NUM_CLASSES = len(CATEGORIES)
IMAGE_SIZE = 64
TRAIN_RATE=0.8

X_train = []
X_test  = []
y_train = []
y_test  = []

# 画像データ用配列
X = []
# ラベルデータ用配列
Y = []

#画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数
def make_sample(files):
    global X, Y
    X = []
    Y = []
    for cat, fname in files:
        add_sample(cat, fname)
    return np.array(X), np.array(Y)

#渡された画像データを読み込んでXに格納し、また、画像データに対応するcategoriesのindexをY格納する関数
def add_sample(cat, fname):
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    data = np.asarray(img)
    # X.append(data)
    # Y.append(cat)
    # データの重増しとラベル付け
    for angle in range(-20, 20, 5):
        img_r = img.rotate(angle)
        data = np.asarray(img_r)
        X.append(data)
        Y.append(cat)
        img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT)
        data = np.asarray(img_trains)
        X.append(data)
        Y.append(cat)

#全データ格納用配列
allfiles = []

#カテゴリ配列の各値と、それに対応するindexを認識し、全データをallfilesにまとめる
for index, classlabel in enumerate(CATEGORIES):
    photos_dir = "./face_img/" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for f in files:
        allfiles.append((index, f))

#シャッフル後、学習データと検証データに分ける
random.shuffle(allfiles)
th = math.floor(len(allfiles) * TRAIN_RATE)
train = allfiles[0:th]
test  = allfiles[th:]
X_train, y_train = make_sample(train)
X_test, y_test = make_sample(test)

print("学習データの数は" + str(len(X_train)))
print("検証データの数は" + str(len(X_test)))
print(y_train[0:500])

X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

# データセットの確認
for i in range(0, 4):
    print("学習データのラベル：", y_train[i])
    plt.subplot(2, 2, i+1)
    plt.axis('off')
    plt.imshow(X_train[i])
plt.show()

xy = (X_train, X_test, y_train, y_test)
np.save("./dataset.npy", xy)