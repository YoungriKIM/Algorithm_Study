# 숫자 앞에 0으로 채워서 가져오고
# npy로 저장해서 쓰자
# >>
# 이미지 128로 줄이기(전처리 후에)

# ==================================================

import PIL.Image as pilimg
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras import preprocessing
from cv2 import resize
import matplotlib.pyplot as plt

### 50000개만 테스트로 해보기 ###

# 이미지 불러오기  =====================================

df_train = []
number = 10

for a in np.arange(0, number):             
    file_path = 'D:/aidata/dacon12/dirty_mnist_2nd/' + str(a).zfill(5) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_train.append(pix)

x = pd.concat(df_train)
x = x.values
# 원래 사이즈는 (256,256)

print(type(x))  # <class 'numpy.ndarray'>
print(x.shape)  # (25600, 256)


