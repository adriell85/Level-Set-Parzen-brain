import cv2
import numpy as np

from ParzenWindow import ParzenWindow as pw

debug = True


def getInitialPointLung(img, cont):
    img = cv2.resize(img, (210, 210))
    if debug:
        cv2.imshow('Input image', img)

    ########################################
    # 1 - Normalizacao
    ########################################
    img_norm = np.zeros_like(img)
    cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
    img_norm = cv2.convertScaleAbs(img_norm)

    ########################################
    # Pré tratamento da imagem - Pulmao
    ########################################

    # Filtro do cv.sepFilter2D
    # filterSep = np.ones_like(img_norm)
    # cv2.sepFilter2D(img_norm, filterSep, kernelX=(3,3), kernelY=(3,3), (-1,-1))
    # background_gaussian = cv2.getGaussianKernel(cv2.gaussian_kernel_size, -1, cv2.CV_32F)
    # filterSep = cv2.sepFilter2D(img_norm, cv2.CV_32F, (5,5), (5,5))

    # clipe = img_norm
    # clipe[clipe < 71] = 0
    # if debug:
    # cv2.imshow('Clipe contraste', clipe)
    # cv2.waitKey(0)
    # exit(0)
    ########################################
    # Aplicaçao do Parzen - Pulmao
    ########################################

    parzen_1 = pw(img_norm, inf_thresh=0, sup_thresh=67, h=0.7, n_points=15, lesion=[], background=[])
    lesao = parzen_1.segmentation()
    if debug:
        cv2.imshow('Entrada para a deteccao das componentes', lesao)

    ########################################
    # Pós tratamento para obter a inicialização do GAC - Pulmao
    ########################################

    maior_comp_1 = pw.maior_comp(lesao)
    maior_comp_1 = ~maior_comp_1

    h, w = lesao.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Aplicação da função Floodfill para limpar ao maximo a imagem de ruidos ou afins
    # Floodfill from point (0, 0)
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
    for i in range(maior_comp_1.shape[1]):
        for j in range(maior_comp_1.shape[0]):
            if 40 < i < 165:
                continue
            maior_comp_1[i, j] = 0
    # if cont==2:
    for i in range(maior_comp_1.shape[1]):
        for j in range(195, 210):
            maior_comp_1[i, j] = 0
    else:
        if cont == 17:
            for i in range(maior_comp_1.shape[1]):
                for j in range(160, 210):
                    maior_comp_1[i, j] = 0

    if debug:
        cv2.imshow('Maior comp 1', maior_comp_1)

    no_traq = ~maior_comp_1
    traq = maior_comp_1
    del maior_comp_1

    ########################################
    # Traqueia
    for i in range(traq.shape[1]):
        for j in range(traq.shape[0]):
            if 90 < i < 120 and 80 < j < 140:
                continue
            traq[j, i] = 0
    if debug:
        cv2.imshow('Traqueia', traq)

    no_traq = ~no_traq
    no_traq = no_traq - traq

    if debug:
        cv2.imshow('Sem a traqueia', no_traq)

    if debug:
        cv2.waitKey(0)
    ########################################

    no_traq = cv2.resize(no_traq, (512, 512))
    return no_traq
