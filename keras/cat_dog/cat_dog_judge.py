import keras
import sys, os
import numpy as np
from keras.models import load_model
from PIL import Image
import glob

imsize = (32, 32)

keras_param = "./cat_dog_cnn.h5"
files = glob.glob("./cat_dog_img/*")

def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img

if __name__ == "__main__":
    result_data = []
    for file in files:
        model = load_model(keras_param)
        img = load_image(file)
        prd = model.predict(np.array([img]))
        print(prd) # 精度の表示
        prelabel = np.argmax(prd, axis=1)
        if prelabel == 0:
            print(">>> 犬")
        elif prelabel == 1:
            print(">>> 猫")