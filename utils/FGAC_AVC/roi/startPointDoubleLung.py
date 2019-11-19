import numpy as np
import cv2
from ParzenWindow import ParzenWindow as pw

debug = False


def maior_comp(img):
    connectivity = 4
    output = cv2.connectedComponentsWithStats(img, 4, cv2.CV_8U)
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    #########################################################
    img_max_ = np.zeros(img.shape, img.dtype)
    img_max_2 = np.zeros(img.shape, img.dtype)

    largecomponent1 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    largecomponent2 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()

    stats[largecomponent2, cv2.CC_STAT_AREA] = largecomponent2
    largecomponent2 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()

    img_max_[labels == largecomponent1] = 255
    img_max_2[labels == largecomponent2] = 255

    image_max = img_max_ + img_max_2

    return img_max_


def getInitialPointLung(img, cont):
    img = cv2.resize(img, (210, 210))
    if debug:
        cv2.imshow('Input image', img)

    img_norm = np.zeros_like(img)
    # ################################################################################
    # 1 - Normalizacao
    # ################################################################################

    cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
    img_norm = cv2.convertScaleAbs(img_norm)

    # ################################################################################
    # Inicio do GAC - Pulmao
    # ################################################################################

    parzen_1 = pw(img_norm, inf_thresh=0, sup_thresh=67, h=0.8, n_points=15)
    lesao = parzen_1.segmentation()
    if debug:
        cv2.imshow('Entrada para a deteccao das componentes', lesao)

    maior_comp_1 = pw.maior_comp(lesao)

    maior_comp_1 = ~maior_comp_1

    h, w = lesao.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # # Floodfill from point (0, 0)
    cv2.floodFill(maior_comp_1, mask, (0, 104), 0)
    cv2.floodFill(maior_comp_1, mask, (104, 0), 0)
    cv2.floodFill(maior_comp_1, mask, (104, 0), 0)
    cv2.floodFill(maior_comp_1, mask, (104, 209), 0)

    maior_comp_1 = ~maior_comp_1

    cv2.floodFill(maior_comp_1, mask, (0, 0), 255)
    cv2.floodFill(maior_comp_1, mask, (0, 209), 255)
    cv2.floodFill(maior_comp_1, mask, (209, 0), 255)
    cv2.floodFill(maior_comp_1, mask, (209, 209), 255)

    maior_comp_1 = ~maior_comp_1
    # Inferior
    for i in range(maior_comp_1.shape[1]):
        for j in range(165, 210):
            maior_comp_1[j, i] = 0

    if cont == 2:
        print(cont)
        for i in range(maior_comp_1.shape[1]):
            for j in range(195, 210):
                maior_comp_1[i, j] = 0
    else:
        if cont == 17:
            print(cont)
            for i in range(maior_comp_1.shape[1]):
                for j in range(160, 210):
                    maior_comp_1[i, j] = 0

    # Superior
    for i in range(maior_comp_1.shape[1]):
        for j in range(0, 40):
            maior_comp_1[j, i] = 0

    if debug:
        cv2.imshow('Maior comp 1', maior_comp_1)

    no_traq = ~maior_comp_1
    traq = maior_comp_1
    del maior_comp_1

    ########################################
    # Traqueia
    for i in range(traq.shape[1]):
        for j in range(0, 90):
            traq[i, j] = 0

    for i in range(traq.shape[1]):
        for j in range(120, 210):
            traq[i, j] = 0

    traq_new = traq
    del traq

    # Inferior
    for i in range(traq_new.shape[1]):
        for j in range(140, 210):
            traq_new[j, i] = 0

    # Superior
    for i in range(traq_new.shape[1]):
        for j in range(0, 80):
            traq_new[j, i] = 0

    if debug:
        cv2.imshow('Traqueia', traq_new)

    no_traq = ~no_traq
    no_traq = no_traq - traq_new

    if debug:
        cv2.imshow('Sem a traqueia', no_traq)

    if debug:
        cv2.waitKey(0)
    ########################################

    no_traq = cv2.resize(no_traq, (512, 512))
    return no_traq