import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np

lady_bin_path = pathlib.Path('../../assets/hw3/ladybin.sec')

lady = np.fromfile(lady_bin_path, dtype=np.uint8).reshape(256, 256)
min_ori = np.min(lady)
max_ori = np.max(lady)

stretched_img = np.uint8(((lady - min_ori) / (max_ori - min_ori)) * 255)

fig, axs = plt.subplots(2, 2, figsize=(12, 6))

axs[0, 0].imshow(cv2.cvtColor(lady, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('before')

axs[0, 1].imshow(cv2.cvtColor(stretched_img, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('after')

hist_ori = cv2.calcHist([lady], [0], None, [256], [0, 256])
axs[1, 0].bar(np.arange(256), hist_ori.ravel())

hist_stretched = cv2.calcHist([stretched_img], [0], None, [256], [0, 256])
axs[1, 1].bar(np.arange(256), hist_stretched.ravel())

plt.show()
