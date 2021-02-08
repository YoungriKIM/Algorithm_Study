# pilimg_1_x_save_npy 에서 저장한 500개의 npy 불러와서 모델 만들기

# 부를 때
# x = np.load('../data/npy/iris_x_data.npy')
# ==================================================

import PIL.Image as pilimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# npy 불러와서 x로 지정 + scaling =====================================
x = np.load('../data/npy/dirty_mnist_train_all.npy').astype('float32')
x = np.where((x<=252)&(x!=0),0.,x)/255.

print(x.shape)  #(500, 256, 256, 1)

# y 불러와서 npy로 변환 =====================================
y = pd.read_csv('D:/aidata/dacon12/dirty_mnist_2nd/dirty_mnist_2nd_answer.csv', index_col=0)[:500].values
print(y.shape)  #(500, 26)
print(type(y))  #<class 'numpy.ndarray'>

# x 전처리 =====================================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=311)

# (100, 256, 256, 1)
# (400, 256, 256, 1)
# (100, 26)
# (400, 26)

# multi classifier model =====================================
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Conv2D(filters = 64, input_shape=(n_inputs.shape[1:]), kernel_size=8, strides= 1, padding='same', kernel_initializer='he_uniform', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 4, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_outputs.shape[1], activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

model = get_model(x_train, y_train)

stop = EarlyStopping(monitor='val_loss', patience=16, mode='auto')

model.fit(x_train, y_train, epochs = 500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[stop])

# 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss, acc: ', loss, acc)


# predict 엑셀로 저장 =====================================
# all_train 불러오기
all_test = np.load('../data/npy/dirty_mnist_test_all.npy').astype('float32')
all_test = np.where((all_test<=252)&(all_test!=0),0.,all_test)/255.

y_pred = model.predict(all_test)
print(y_pred.shape) #(500, 26)
print(np.min(y_pred), np.max(y_pred))   #0.433219 0.5084275

subfile = pd.read_csv('D:/aidata/dacon12/submission/sample_submission.csv')

y_pred = pd.DataFrame(y_pred.round(2))
print(y_pred.head())

# subfile.iloc[:500, 1:] = y_pred

# print(subfile.head())

y_pred.to_csv('../data/csv/dacon12/subsave/p2_0209.csv', index=False)
print('===== save complete =====')


# ===============================
# p2_0209.csv > dacon score : 	0.48086 > 단단히 잘못되고 있눈..너낌~^^