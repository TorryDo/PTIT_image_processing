import pathlib

import matplotlib.pyplot as plt
import numpy as np

mammogram_bin_path = pathlib.Path('../../assets/hw3/Mammogrambin.sec')

data = np.fromfile(mammogram_bin_path, dtype=np.uint8, count=256 * 256)
image = data.reshape(256, 256)
threshold_value = 128
binary_image = (image > threshold_value).astype(np.uint8) * 255

plt.imshow(binary_image, cmap='gray')
plt.title("Binary Image")
plt.show()


# b, Approximate Contour Image
def approximate_contour(bin_img):
    ct_image = np.zeros_like(bin_img)

    for y in range(1, bin_img.shape[0] - 1):
        for x in range(1, bin_img.shape[1] - 1):
            if bin_img[y, x] == 255:
                neighbors = [
                    bin_img[y - 1, x],
                    bin_img[y - 1, x + 1],
                    bin_img[y, x + 1],
                    bin_img[y + 1, x + 1],
                    bin_img[y + 1, x],
                    bin_img[y + 1, x - 1],
                    bin_img[y, x - 1],
                    bin_img[y - 1, x - 1]
                ]

                if 0 in neighbors:
                    ct_image[y, x] = 255

    return ct_image


contour_image = approximate_contour(binary_image)

plt.imshow(contour_image, cmap='gray')
plt.title("Contour Image")
plt.show()
