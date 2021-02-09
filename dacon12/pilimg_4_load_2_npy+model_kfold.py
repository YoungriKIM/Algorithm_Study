# 이전 파일 가져와서 kfold 적용 해보기 !

import PIL.Image as pilimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam


# npy 불러와서 x로 지정 + scaling =====================================
x = np.load('../data/npy/dirty_mnist_train_all(10000).npy').astype('float32')
x[x < 253] = 0
x = x/255.

print(x.shape)

# y 불러와서 npy로 변환 =====================================
y = pd.read_csv('D:/aidata/dacon12/dirty_mnist_2nd/dirty_mnist_2nd_answer.csv', index_col=0)[:10000].values
print(y.shape)  #(500, 26)
print(type(y))  #<class 'numpy.ndarray'>


# x 전처리 =====================================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=311)

# (n, 256, 256, 1)
# (n, 256, 256, 1)
# (n, 26)
# (n, 26)


# multi classifier model =====================================
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Conv2D(filters = 64, input_shape=(n_inputs.shape[1:]), kernel_size=4, strides= 1, padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_outputs.shape[1], activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])
    return model

# model = get_model(x_train, y_train)

# kfold정의
kfold = KFold(n_splits=5, shuffle=True)

# scores = cross_val_score(model, x_train, y_train, cv=kfold)
# print('scores: ', scores)

stop = EarlyStopping(monitor='val_loss', patience=16, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')


for train_index, test_index in kfold.split(x_train):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = get_model(x_train, y_train)

    model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, callbacks=[stop], validation_split=0.2)

    loss, acc = model.evaluate(x_test, y_test)
    print('loss, acc: ', loss, acc)


# =============================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
# =============================================

ing