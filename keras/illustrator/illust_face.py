import cv2
import glob
import os

# import urllib.request
# url = 'https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml'
# with urllib.request.urlopen(url) as u:
#   with open('lbpcascade_animeface.xml', 'bw') as o:
#     o.write(u.read())

CATEGORIES = ["ponkan", "ixy", "makihituzi", "tokunou", "anmi"]
NUM_CLASSES = len(CATEGORIES)

# 元画像を取り出して顔部分を正方形で囲み、64×64pにリサイズ、別のファイルにどんどん入れてく
for index, classlabel in enumerate(CATEGORIES):
    out_dir = "./face_img/" + classlabel
    photos_dir = "./illust_img/" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    num = 0

    for f in files:
        image = cv2.imread(str(f))
        if image is None:
            print("Not open:")
            continue
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
                    num += 1
                else:
                    image = cv2.imread(str(f))
        else:
            print("no face")
            continue
        print(image.shape)
        # 保存
        fileName = os.path.join(out_dir, str(num) + ".jpg")
        cv2.imwrite(str(fileName), image)
