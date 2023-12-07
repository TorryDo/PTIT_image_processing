import cv2

# a, help
help(cv2.imread)
help(cv2.imwrite)

# b, obtain the image
# c, read the image
lenagray_img_path = '../../assets/hw2/lenagray.jpg'
J1 = cv2.imread(lenagray_img_path)

# reverse img pixel
J2 = 255 - J1

# show the images
cv2.imshow('J2', J2)
cv2.imwrite('J2.jpg', J2)
cv2.waitKey(0)
cv2.destroyAllWindows()