# 숫자 앞에 0으로 채워서 가져오고
# npy로 저장해서 쓰자

# npy로 저장할 때 쓰는 것 ===========================
# np.save('../data/npy/iris_x_data.npy', arr=x_data)    # 저장 할 때는 npy로 한다.
# 부를 때
# x = np.load('../data/npy/iris_x_data.npy')
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

# 이미지 전처리 -------------------------------------

#254보다 작고 0이아니면 0으로 만들어주기
x_df2 = np.where((x <= 254) & (x != 0), 0, x)

# 이미지 팽창
x_df3 = cv2.dilate(x_df2, kernel=np.ones((2, 2), np.uint8), iterations=1)

# 블러 적용, 노이즈 제거
x_df4 = cv2.medianBlur(src=x_df3, ksize= 5)

cv2.imshow('1', x_df4)
cv2.waitKey()

# 이미지 리쉐잎 -------------------------------------
# 리쉐잎
x_dataset = x_df4.reshape(number, 256, 256, 1)

print(x_dataset.shape)

cv2.imshow('2', x_dataset[0])
cv2.waitKey()

# npy저장  =====================================
# npy 한개 용량 65Kb
# > 50,000개 일 떄 : 3.25Gb
np.save('../data/npy/dirty_mnist_train_all(50000)_small.npy', arr=x_dataset)
print('===== save complete =====')

# 그냥 500개 저장한 용량: 32Mb      > 그냥 저장하고 불러와서 전처리 하기로
# 500 + scaling(나누기255) 용량 : 256MB
# 500 + scaling(나누기255) + 특정 값 이하 0 수렴 용량: 256Mb
# 500 + 특정 값 이하 0 수렴 용량: 256Mb

# =====================================
# 저장 후에 load 파일 만들어서 모델 돌려보기
# 50000개 저장 완료