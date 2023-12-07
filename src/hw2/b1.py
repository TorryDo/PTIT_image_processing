import numpy as np
import matplotlib.pyplot as plt
import cv2

lenabin_path = '../../assets/hw2/lenabin.sec'
peppersbin_path = '../../assets/hw2/peppersbin.sec'

# a, read and display the images
# b, define a new 256x256 img J
# c, define a new 256x256 img K
J = np.fromfile(lenabin_path, dtype='uint8')
J = J.astype(int)
J = J.reshape(256, 256)
cv2.imwrite('lena2.jpg', J)

K = np.fromfile(peppersbin_path, dtype='uint8')
K = K.astype(int)
K = K.reshape(256, 256)
cv2.imwrite('peppersbin2.jpg', K)

# d, show the images J and K
plt.imshow(J, cmap='gray')
plt.show()

plt.imshow(K, cmap='gray')
plt.show()