import cv2

lena512color_img_path = '../../assets/hw2/lena512color.jpg'

J1 = cv2.imread(lena512color_img_path)

J2 = J1.copy()
J2[:, :, 0] = J1[:, :, 2]  # Red band of J2 = Blue band of J1
J2[:, :, 2] = J1[:, :, 0]  # Blue band of J2 = Red band of J1

cv2.imshow('J2', J2)
cv2.imwrite('J2_color.jpg', J2)
cv2.waitKey(0)
cv2.destroyAllWindows()