# npy 저장한거 잘 됐는지 확안해보자
# 샘플로 500개만 했음

import numpy as np
import pandas as pd


'''
# x train ====
x = np.load('../data/npy/dirty_mnist_train_all.npy').astype('float32')/255
x = np.where((x<=251/255)&(x!=0),0.,x)

print(x.shape)
print(x[0])
'''

# y train ====
y = pd.read_csv('D:/aidata/dacon12/dirty_mnist_2nd/dirty_mnist_2nd_answer.csv', index_col=0)[:500].values
print(y.shape)  #(500, 26)
print(type(y))  #<class 'numpy.ndarray'>
print(y[0])