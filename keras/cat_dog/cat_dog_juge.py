import keras
import sys, os
import numpy as np
from keras.models import load_model
from PIL import Image

imsize = (64, 64)

testpic     = "./06.jpg"
keras_param = "./cat_dog_cnn.h5"

def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img

if __name__ == "__main__":

    model = load_model(keras_param)
    img = load_image(testpic)
    prd = model.predict(np.array([img]))
    print(prd) # 精度の表示
    prelabel = np.argmax(prd, axis=1)
    if prelabel == 0:
        print(">>> 犬")
    elif prelabel == 1:
        print(">>> 猫")