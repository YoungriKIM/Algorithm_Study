## predict에 쓸!!! all_test npy 저자용ㅇ!!!

# 숫자 앞에 0으로 채워서 가져오고
# npy로 저장해서 쓰자

# npy로 저장할 때 쓰는 것 ===========================
# np.save('../data/npy/iris_x_data.npy', arr=x_data)    # 저장 할 때는 npy로 한다.
# 부를 때
# x = np.load('../data/npy/iris_x_data.npy')
# ==================================================

import PIL.Image as pilimg
import numpy as np
import pandas as pd

### 500개만 테스트로 해보기 ###

# 이미지 불러오기  =====================================

df_train = []

for a in np.arange(50000, 50500):
    b = str(a)
    c = str('0'*(5-len(b)))
    i = (c+b)               
    file_path = 'D:/aidata/dacon3/test_dirty_mnist_2nd/' + str(i) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_train.append(pix)

x = pd.concat(df_train)
x = x.values
# 원래 사이즈는 (256,256)

print(type(x))  #<class 'numpy.ndarray'>
print(x.shape)

# 리쉐잎
number = 500
x_dataset = x.reshape(number, 256, 256, 1)

# npy 저장 전 전처리  =====================================
# 252아래 특성 0으로 수렴하고 scaling
# x_dataset = np.where((x_dataset<=252)&(x_dataset!=0),0.,x_dataset)

print(x_dataset.shape)      # (500, 256, 256, 1)
# print(x_dataset)


# npy저장  =====================================
# npy 한개 용량 65Kb
# > 50,000개 일 떄 : 3.25Gb
np.save('../data/npy/dirty_mnist_test_all.npy', arr=x_dataset)
print('===== save complete =====')

# 그냥 500개 저장한 용량: 32Mb      > 그냥 저장하고 불러와서 전처리 하기로
# 500 + scaling(나누기255) 용량 : 256MB
# 500 + scaling(나누기255) + 특정 값 이하 0 수렴 용량: 256Mb
# 500 + 특정 값 이하 0 수렴 용량: 256Mb

# =====================================
# 저장 후에 load 파일 만들어서 모델 돌려보기