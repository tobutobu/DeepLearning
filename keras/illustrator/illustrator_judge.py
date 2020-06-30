import numpy as np
from keras.models import load_model
from PIL import Image
import glob
import cv2
import matplotlib.pyplot as plt

imsize = (64, 64)

keras_param = "./illustrator_cnn.h5"
files = glob.glob("./img/*")

def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        print("Not open:")
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_file = "lbpcascade_animeface.xml"
    cascade = cv2.CascadeClassifier(cascade_file)
    # 顔認識の実行
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    if len(face_list) > 0:
        for rect in face_list:
            x, y, width, height = rect
            image = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            if image.shape[0] >= 64:
                image = cv2.resize(image, (64, 64))
            else:
                image = cv2.imread(str(path))
    else:
        print("no face")
    print(image.shape)
    # cvからpilに変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    image = np.asarray(image)
    image = image / 255.0
    return image

if __name__ == "__main__":
    # 画像認識
    num = 0
    num1 = 0
    num2 = 0
    num3 = 0
    num4 = 0

    for file in files:
        model = load_model(keras_param)
        img = load_image(file)
        prd = model.predict(np.array([img]))
        img = Image.open(file)
        print(prd) # 精度の表示
        prelabel = np.argmax(prd, axis=1)
        if prelabel == 0:
            num += 1
            fileName = str(num) + ".jpg"
            img.save("./img_sort/ponkan~/" + str(fileName))
            print(">>> ぽんかん⑧")
        elif prelabel == 1:
            num1 += 1
            fileName = str(num1) + ".jpg"
            img.save("./img_sort/ixy~/" + str(fileName))
            print(">>> ixy")
        elif prelabel == 2:
            num2 += 1
            fileName = str(num2) + ".jpg"
            img.save("./img_sort/makihituzi~/" + str(fileName))
            print(">>> 巻羊")
        elif prelabel == 3:
            num3 += 1
            fileName = str(num3) + ".jpg"
            img.save("./img_sort/tokunou~/" + str(fileName))
            print(">>> 得能")
        elif prelabel == 4:
            num4 += 1
            fileName = str(num4) + ".jpg"
            img.save("./img_sort/anmi~/" + str(fileName))
            print(">>> anmi")

    # データ確認
    # for i, file in enumerate(files):
    #     img = load_image(file)
    #     plt.imshow(img)
    #     plt.subplot(3, 3, i + 1)
    #     plt.axis('off')
    # plt.show()