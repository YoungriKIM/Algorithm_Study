# pilimg_1_x_save_npy 에서 저장한 500개의 npy 불러와서 모델 만들기

# 부를 때
# x = np.load('../data/npy/iris_x_data.npy')
# ==================================================

import PIL.Image as pilimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 5만개 전체로 진행 (256,256)

# npy 불러와서 지정 =====================================
x = np.load('../data/npy/dirty_mnist_train_all(50000).npy')[:100]
x = x.reshape(100,256,256)

# y 불러와서 지정 =====================================
y = np.load('../data/npy/dirty_mnist_2nd_answer.npy')[:100]

# x 전처리 =====================================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# multi classifier model =====================================
# def get_model(n_inputs, n_outputs):
#     model = Sequential()
#     model.add(Conv2D(filters = 32, input_shape=(n_inputs.shape[1:]), kernel_size=5, strides= 1, padding='same', kernel_initializer='he_uniform', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(32, 2, padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Flatten())
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(n_outputs.shape[1], activation='sigmoid'))

#     model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.02), metrics=['acc'])
#     return model

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(LSTM(64, input_shape=(n_inputs.shape[1:]), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    # model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_outputs.shape[1], activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.02), metrics=['acc'])
    return model

model = get_model(x_train, y_train)

stop = EarlyStopping(monitor='val_loss', patience=16, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=8, mode='min')
file_path = '../data/modelcheckpoint/dacon12/p5_1_0211.hdf5'
mc = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True, mode='min')

model.fit(x_train, y_train, epochs = 1000, batch_size=3, verbose=1, validation_split=0.2, callbacks=[stop, lr, mc])

# 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss, acc: ', loss, acc)


# predict 엑셀로 저장 =====================================
# all_train 불러오기
all_test = np.load('../data/npy/dirty_mnist_test_all(5000).npy')[:100]
all_test = all_test.reshape(100,256,256)

# 예측
y_pred = model.predict(all_test)
print(y_pred.shape) #(500, 26)
print(np.min(y_pred), np.max(y_pred))   #0.433219 0.5084275

subfile = pd.read_csv('D:/aidata/dacon12/submission/sample_submission.csv')

y_pred = pd.DataFrame(y_pred.round(2))
print(y_pred.head())

# subfile.iloc[:500, 1:] = y_pred

# print(subfile.head())

y_pred.to_csv('../data/csv/dacon12/subsave/p5_0211_1.csv', index=False)
print('===== save complete =====')


# ===============================
# p2_0209.csv > dacon score : 	0.48086 > 단단히 잘못되고 있눈..너낌~^^

# p2_0209-2.csv
# loss, acc:  0.6902444362640381 0.0 > dacon score: 0.5334307692

# p5_0211_1
# 모델 lstm으로 수정
# loss: 0.6837 - acc: 0.0 비슷비슷.. 토치를 독학해야겠음 