from PIL import Image
import os, glob
import numpy as np
from PIL import ImageFile
import random, math
# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

CATEGORIES = ["dogs", "cats"]
NUM_CLASSES = len(CATEGORIES)
IMAGE_SIZE = 64
NUM_TESTDATA = 25
TRAIN_RATE=0.8

X_train = []
X_test  = []
y_train = []
y_test  = []

for index, classlabel in enumerate(CATEGORIES):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        data = np.asarray(image)
        # データの重増しとラベル付け
        for angle in range(-20, 20, 5):
            img_r = image.rotate(angle)
            data = np.asarray(img_r)
            X_train.append(data)
            y_train.append(index)
            img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT)
            data = np.asarray(img_trains)
            X_train.append(data)
            y_train.append(index)

# 学習データと検証データに分ける
th = math.floor(len(X_train) * TRAIN_RATE)
X_test = X_train[th:]
y_test = y_train[th:]
X_train = X_train[0:th]
y_train = y_train[0:th]

print("学習データの数は" + str(len(X_train)))
print("検証データの数は" + str(len(X_test)))

X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

xy = (X_train, X_test, y_train, y_test)
np.save("./dataset.npy", xy)