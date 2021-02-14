# https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/
# 4.2 Adapted Algorithm 적용


import PIL.Image as pilimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import warnings
# warnings.filterwarnings('ignore')

# npy 불러와서 x 지정 =====================================
x = np.load('../data/npy/dirty_mnist_train_all(50000).npy')[:100]
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

print(x.shape)      #(100, 65536)


# y 불러와서 지정 =====================================
y = np.load('../data/npy/dirty_mnist_2nd_answer.npy')[:100]

print(y.shape)      #(100, 26)


# 전처리 =====================================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


# 모델 구성 =====================================
from skmultilearn.adapt import MLkNN

classifier = MLkNN({'k':26})

# fit =====================================
classifier.fit(x_train, y_train)

# predict =====================================
y_pred = classifier.predict(x_test)

accuracy_score(y_test, y_pred)
print('accuracy_score: ', accuracy_score)


# error  ---------------------------------------------
# FutureWarning: Pass n_neighbors=26 as keyword args. From version 0.25 passing these as positional arguments will result in an error
#   warnings.warn("Pass {} as keyword args. From version 0.25 "

# TypeError: '<' not supported between instances of 'dict' and 'int'

# 오버스톡에 질문 올림
# https://stackoverflow.com/questions/66196888/error-typeerror-not-supported-between-instances-of-dict-and-int-usin