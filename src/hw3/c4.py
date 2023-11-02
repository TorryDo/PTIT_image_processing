import pathlib

import matplotlib.pyplot as plt
import numpy as np
import cv2

johnny = pathlib.Path('../../assets/hw3/johnnybin.sec')


john_data = np.fromfile(johnny, dtype=np.uint8)
john_image = john_data.reshape(256, 256)

equalized_image = cv2.equalizeHist(john_image)

fix, axs = plt.subplots(2, 2, figsize=(12,6))

axs[0, 0].set_title('before')
axs[0, 0].imshow(john_image, cmap="gray")
axs[0, 0].axis('off')

axs[0, 1].set_title('after')
axs[0, 1].imshow(equalized_image, cmap="gray")
axs[0, 1].axis('off')

axs[1, 0].set_title('before')
axs[1, 0].hist(john_image.ravel(), 256, [0, 256])
axs[1, 0].set_xlim([0, 256])

axs[1, 1].set_title('after')
axs[1, 1].hist(equalized_image.ravel(), 256, [0, 256])
axs[1, 1].set_xlim([0, 256])

plt.show()
