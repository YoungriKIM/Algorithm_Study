# https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/
# 4.1.1 Binary Relevance 적용

import PIL.Image as pilimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
# import warnings
# warnings.filterwarnings('ignore')

# npy 불러와서 x 지정 =====================================
x = np.load('../data/npy/dirty_mnist_train_all(50000).npy')[:100]
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
print(x.shape)


# y 불러와서 지정 =====================================
y = np.load('../data/npy/dirty_mnist_2nd_answer.npy')[:100]

print(y.shape)


# 전처리 =====================================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


'''
# 모델 구성 =====================================
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
# classifier = BinaryRelevance(GaussianNB())

classifier = BinaryRelevance(
            classifier = SVC(),
            require_dense = [False, True]
        )

# fit =====================================
classifier.fit(x_train, y_train)

# predict =====================================
y_pred = classifier.predict(x_test)

accuracy_score(y_test, y_pred)
print('accuracy_score: ', accuracy_score)



# error -------------------------------------------------
# accuracy_score:  <function accuracy_score at 0x00000286079E1EE0>
'''

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

parameters = [
    {
        'classifier': [MultinomialNB()],
        'classifier__alpha': [0.7, 1.0],
    },
    {
        'classifier': [SVC()],
        'classifier__kernel': ['rbf', 'linear'],
    },
]


clf = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy')
clf.fit(x_train, y_train)

print (clf.best_params_, clf.best_score_)