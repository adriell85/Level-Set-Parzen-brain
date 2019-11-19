from cv2 import imread, imwrite

for i in range(1, 37):
    img = imread('binary_images/doctor_images/{}.png'.format(i))
    img[img == 1] = 255
    img[img == 0] = 0
    imwrite('binary_images/doctor_images_255/{}.png'.format(i), img)
