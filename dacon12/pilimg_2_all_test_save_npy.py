## predict에 쓸!!! all_test npy 저자용ㅇ!!!

# 숫자 앞에 0으로 채워서 가져오고
# npy로 저장해서 쓰자

# npy로 저장할 때 쓰는 것 ===========================
# np.save('../data/npy/iris_x_data.npy', arr=x_data)    # 저장 할 때는 npy로 한다.
# 부를 때
# x = np.load('../data/npy/iris_x_data.npy')
# ==================================================

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import PIL.Image as pilimg

### 전체(5000)로 저장 ###

# x train 데이터 불러오기 -------------------------------------
df_pix = []
number = 55000

for a in np.arange(50000, number):             
    file_path = 'D:/aidata/dacon12/test_dirty_mnist_2nd/' + str(a) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_pix.append(pix)

x_df = pd.concat(df_pix)
x_df = x_df.values
# 원래 사이즈는 (256,256)

print(type(x_df))  # <class 'numpy.ndarray'>
print(x_df.shape)  # (25600, 256)

# 이미지 전처리 -------------------------------------

#254보다 작고 0이아니면 0으로 만들어주기
x_df2 = np.where((x_df <= 254) & (x_df != 0), 0, x_df)

# 이미지 팽창
x_df3 = cv2.dilate(x_df2, kernel=np.ones((2, 2), np.uint8), iterations=1)

# 블러 적용, 노이즈 제거
x_df4 = cv2.medianBlur(src=x_df3, ksize= 5)


# 이미지 리쉐잎 -------------------------------------
# 리쉐잎
x_dataset = x_df4.reshape(5000, 256, 256, 1)

print(x_dataset.shape)

# npy로 저장 -------------------------------------
np.save('../data/npy/dirty_mnist_test_all(5000).npy', arr=x_dataset)
print('===== save complete =====')

# npy로 저장 잘 되었나 확인 -------------------------------------
load_x = np.load('../data/npy/dirty_mnist_test_all(5000).npy')
print('===== save complete =====')

print(load_x.shape) 


# 저장 확인