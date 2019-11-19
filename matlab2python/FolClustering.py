#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script to load voxel data.
"""
from skimage.feature import local_binary_pattern
from os.path import abspath, join, dirname
import matplotlib.pyplot as ppl
import numpy as np
import cv2
import sys

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from data_information import dcm_information as di
from AreaLabel import AreaLabel


def normalizeImage(v):
    v = (v - v.min()) / (v.max() - v.min())
    result = (v * 255).astype(np.uint8)
    return result


def fol(image_nput, base, count):
    image = image_nput + local_binary_pattern(image_nput, 8, 1, "uniform")
    return np.log(image / np.log(base)).astype(np.int8), count


# lbp_MOD = 0


def clustering_log(img):
    count_debug = 1
    # img_num = 1
    show = False
    # st3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    st3 = np.ones((3, 3), dtype=np.int8)
    log = 1.05
    # img1 = cv2.imread("datasets/cranio/cranio_1.png")
    # img = di.load_dcm("datasets/OriginalCT/Imagem{}.dcm".format(img_num))
    # count_debug = di.show_figures(img, count_debug)
    img = img - 1024

    img = np.where(img < 80, img, 0)
    img = np.where(img > 0, img, 0)

    fol_image, count_debug = fol(img, log, count_debug)
    # TODO 0: A base do algoritmo FoL pode ser aprendida, aplicar aqui o algoritmo de treinamento

    fol_image[fol_image != 7] = 0

    # if show:
    #     count_debug = di.show_figures(fol_image, count_debug)

    # # Image opening
    # for _ in range(9):
    #     fol_open = cv2.dilate(cv2.erode(fol_image.astype(np.uint8), st3), st3)

    output = cv2.connectedComponentsWithStats(fol_image, 4, cv2.CV_8U)
    indice1 = np.argsort(output[2][:, -1])[::-1]
    avc_seg = np.where(output[1] == indice1[1], 1, 0)
    # avc_seg = np.where(output[1] == 8, 1, 0)

    # labels = labels.astype(np.uint8)
    # area = [sum(output[1] == x) for x in range(0, output[0])]  # Number of labels
    # Return valor, indice and L
    # bigg = output[2][:, 4]
    # asort = output[2][output[2][:, 4].argsort()]
    # arr = argsort(output[2], axis=0)

    # arr = argsort(bigg)

    # centroids = output[2]
    # cen = centroids[:, -1]
    # ss = np.sort(centroids[:, -1])
    # ssu = np.sort(np.unique(centroids[:, -1]))
    # asr = np.argsort(centroids[:, -1])
    # asru = np.argsort(np.unique(centroids[:, -1]))

    # count_debug = di.show_figures(output[1], count_debug)
    # count_debug = di.show_figures(avc_seg, count_debug)
    # ppl.show()
    #
    # exit()

    # TODO 1: Selecionar a label que coincide com o local da semente
    # TODO 2: A label escolhida, é a região pré-segmentada, passar essa regiao pro Parzen terminar.

    # cv2.imshow('Blood',normalizeImage( FoLImage>220))
    # cv2.waitKey(0)
    return (2 * avc_seg.astype(np.int16)) - 1