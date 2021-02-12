# y_train 저장용

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import PIL.Image as pilimg


# y 불러와서 numpy화 -------------------------------------
y = pd.read_csv('D:/aidata/dacon12/dirty_mnist_2nd/dirty_mnist_2nd_answer.csv', index_col=0).values

# npy로 저장 -------------------------------------
np.save('../data/npy/dirty_mnist_2nd_answer.npy', arr=y)
print('===== save complete =====')

# npy로 저장 잘 되었나 확인 -------------------------------------
load_y = np.load('../data/npy/dirty_mnist_2nd_answer.npy')
print('===== save complete =====')

print(load_y.shape) 