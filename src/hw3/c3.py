import pathlib

import matplotlib.pyplot as plt
import numpy as np
import cv2

actont_bin_bin_path = pathlib.Path('../../assets/hw3/actontBinbin.sec')


# # Đọc ảnh
# with open(actont_bin_bin_path, 'rb') as f:
#     bin_data = np.fromfile(f, dtype=np.uint8)
#     bin_image = bin_data.reshape(256, 256)
#
# # Đọc hình ảnh mẫu
# template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
#
# # Thực hiện matching
# result = cv2.matchTemplate(bin_image, template, cv2.TM_CCOEFF_NORMED)
#
# # Xác định ngưỡng
# threshold = 0.2  # Giá trị ngưỡng tùy chỉnh
#
# # Tạo hình ảnh kết quả J2
# J2 = np.where(result >= threshold, 255, 0).astype(np.uint8)
#
# #Show ảnh gốc
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Ảnh gốc')
# plt.imshow(bin_image, cmap='gray')
# plt.axis('off')
#
# #Show ảnh J2
# plt.subplot(1, 2, 2)
# plt.title('Kết quả')
# plt.imshow(J2, cmap='gray')
# plt.axis('off')
#
# plt.show()

def binary_template_matching(input_img, templ):
    output_img = np.zeros_like(input_img)

    for i in range(input_img.shape[0] - templ.shape[0] + 1):
        for j in range(input_img.shape[1] - templ.shape[1] + 1):
            output_img[i, j] = np.sum(input_img[i: i + templ.shape[0], j: j + templ.shape[1]] * templ)

    return output_img


def threshold_image(image, threshold):
    binary_image = np.zeros_like(image)
    binary_image[image >= threshold] = 255

    return binary_image


input_image = np.fromfile(actont_bin_bin_path, dtype=np.uint8)
input_image = input_image.reshape((256, 256))

template = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
)

output_image = binary_template_matching(input_image, template)

binary_output_image = threshold_image(output_image, 128)

plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(input_image, cmap="gray")
plt.title("Input image")

plt.subplot(132)
plt.imshow(template, cmap="gray")
plt.title("Template")

plt.subplot(133)
plt.imshow(binary_output_image, cmap="gray")
plt.title("Output image")

plt.show()
