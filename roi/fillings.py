import cv2
import numpy as np

from ParzenWindow import ParzenWindow as pw


def getInitialPointLung(image, cont):
    # image = cv2.resize(image, (150, 150))

    norm_img = np.zeros_like(image)

    cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
    image = cv2.convertScaleAbs(norm_img)
    cv2.imshow('0 - original', image)

    mean, std = cv2.meanStdDev(norm_img)
    image[norm_img < mean] = 0

    del norm_img

    # ret, bin_img = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    parzen_1 = pw(image, inf_thresh=0, sup_thresh=60, h=1, n_points=15)
    bin_img = parzen_1.segmentation()

    thresh_img = 1 - bin_img
    cv2.imshow("1 - Threshold", thresh_img * 255)

    del bin_img

    ee_2x2 = np.ones((2, 2))
    ee_3x3 = np.ones((3, 3))
    # ee_4x4 = np.ones((4, 4))
    ee_9x9 = np.ones((9, 9))

    for i in range(1):
        thresh_img = cv2.dilate(thresh_img, ee_2x2)
        thresh_img = cv2.erode(thresh_img, ee_2x2)

    cv2.imshow('2 - Pos erosao do filtro de Otsu', thresh_img * 255)

    thresh_img = 1 - thresh_img
    cv2.imshow('3 - Inversao de valores', thresh_img * 255)

    connectivity = 8
    output1 = cv2.connectedComponentsWithStats(thresh_img, connectivity, cv2.CV_8U)
    labels1 = output1[1]
    stats1 = output1[2]

    largecomponent1 = 1 + stats1[1:, cv2.CC_STAT_AREA].argmax()
    stats1[largecomponent1, cv2.CC_STAT_AREA] = largecomponent1

    bone = np.zeros(thresh_img.shape, thresh_img.dtype)
    bone[labels1 == largecomponent1] = 1
    cv2.imshow('4 - Encontrar o osso', bone * 255)

    del thresh_img

    bone_inv = 1 - bone

    del bone

    cv2.imshow('5 - Inversao do osso', bone_inv * 255)
    ########################################################

    position = [0, 0]
    for x in range(bone_inv.shape[0]):
        if position[0] != 0 or position[1] != 0:
            break
        for y in range(bone_inv.shape[1]):
            if bone_inv[x, y] == 0:
                position[0] = x
                position[1] = y
                break

    bone_inv[0:position[0] + 10, :] = 0
    ########################################################

    position2 = [0, 0]
    for x in range(bone_inv.shape[0] - 1, 0, -1):
        if position2[0] != 0 or position2[1] != 0:
            break
        for y in range(bone_inv.shape[1] - 1, 0, -1):
            if bone_inv[x, y] == 0:
                position2[0] = x
                position2[1] = y
                break

    bone_inv[position2[0] - 10: 512, :] = 0
    ########################################################

    for x in range(position[0] + 10, 512):
        for y in range(bone_inv.shape[1]):
            if bone_inv[x][y] == 0:
                break
            if bone_inv[x][y] == 1:
                bone_inv[x][y] = 0
    ########################################################

    for x in range(position2[0] - 10, 0, -1):
        for y in range(bone_inv.shape[1] - 1, 0, -1):
            if bone_inv[x][y] == 0:
                break
            if bone_inv[x][y] == 1:
                bone_inv[x][y] = 0

    cv2.imshow('6 - Remover osso', bone_inv * 255)

    for i in range(1):
        bone_inv = cv2.dilate(bone_inv, ee_3x3)
        bone_inv = cv2.erode(bone_inv, ee_3x3)

    cv2.imshow('7 - Fazer abertura na imagem', bone_inv * 255)

    output2 = cv2.connectedComponentsWithStats(bone_inv, connectivity, cv2.CV_8U)
    labels2 = output2[1]
    stats2 = output2[2]

    lungs = np.zeros(bone_inv.shape, bone_inv.dtype)

    largecomponent21 = 1 + stats2[1:, cv2.CC_STAT_AREA].argmax()
    stats2[largecomponent21, cv2.CC_STAT_AREA] = largecomponent21
    largecomponent22 = 1 + stats2[1:, cv2.CC_STAT_AREA].argmax()
    stats2[largecomponent22, cv2.CC_STAT_AREA] = largecomponent22

    lungs[labels2 == largecomponent21] = 1
    lungs[labels2 == largecomponent22] = 1

    cv2.imshow('8 - Dois pulmoes', lungs * 255)

    for i in range(1):
        lungs = cv2.dilate(lungs, ee_9x9)
        lungs = cv2.erode(lungs, ee_9x9)
    cv2.waitKey(0)
    return lungs
