# pilimg로 불러오고 npy로 저장해서 용량 비교하기

import PIL.Image as pilimg
import numpy as np
import pandas as pd
 
# Read image ==================================================
image = pilimg.open('D:/aidata/dacon3/dirty_mnist_2nd/00000.png')
 
# Display image ==================================================
image.show()
 
# Fetch image pixel data to numpy array ==================================================
pix = np.array(image)
print(pix.shape)        #(256, 256)