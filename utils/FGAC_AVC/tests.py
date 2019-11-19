import sys

import numpy as np
from imageio import imread
from matplotlib import pyplot as ppl

sys.path.append('roi')
from roi import startPointDoubleLung as st
import morphsnakes
import cv2


def rgb2gray(img):
    return 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def circle_levelset(shape, center, sqradius, scalerow=1.0):
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T) ** 2, 0))
    u = np.float_(phi > 0)
    return u


def test_GAC(cont, img, p_alpha=1000, p_sigma=6, p_threshold=0.47,
             p_balloon=1, p_smoothing=1, p_num_iters=70
             ):
    gI = morphsnakes.gborders(img, alpha=p_alpha, sigma=p_sigma)
    mgac = morphsnakes.MorphGAC(gI, smoothing=p_smoothing, threshold=p_threshold, balloon=p_balloon)
    mgac.levelset = st.get_initial_point_lung(img, cont)
    ppl.figure()
    img = morphsnakes.evolve_visual(cont, mgac, num_iters=p_num_iters, background=img)
    return img


#######################################
# Retirando apenas o cranio das imagens
#######################################
def new_image(img):
    cv2.imshow(' 1  Input image', img)
    cv2.imwrite("C:/Users/Lapisco02/Desktop/atualizado/extracao do cranio/imagem88.png", img)

    #################################
    # 1 - Normalizacao
    #################################
    img_norm = np.zeros_like(img)
    cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
    img_norm = cv2.convertScaleAbs(img_norm)

    ret, img_bin = cv2.threshold(img_norm, 240, 255, cv2.THRESH_BINARY)

    cv2.imshow(' 2 - Image bin', img_bin)
    cv2.imwrite("C:/Users/Lapisco02/Desktop/atualizado/extracao do cranio/1_thresh.png", img_bin)

    #########################################################
    connectivity = 4
    output = cv2.connectedComponentsWithStats(img_bin, 4, cv2.CV_8U)
    labels = output[1]  # AM: Rotulo das componentes
    stats = output[2]  # AM: Estatistica das componentes
    centroids = output[3]  # AM: Centroids das componentes
    #########################################################

    img_max_ = np.zeros(img_bin.shape, img_bin.dtype)
    img_max_2 = np.zeros(img_bin.shape, img_bin.dtype)

    largecomponent1 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    largecomponent2 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()

    stats[largecomponent1, cv2.CC_STAT_AREA] = largecomponent1
    stats[largecomponent2, cv2.CC_STAT_AREA] = largecomponent2

    img_max_[labels == largecomponent1] = 255
    img_max_2[labels != largecomponent2] = 255

    cv2.imshow(' 3 - Apenas o osso do cranio', img_max_)
    cv2.imwrite("C:/Users/Lapisco02/Desktop/atualizado/extracao do cranio/2_osso_ccranio.png", img_max_)

    # Diferença
    dif = img - img_max_
    cv2.imshow(' 4 - Diferenca 1 e 3', dif)
    cv2.imwrite("C:/Users/Lapisco02/Desktop/atualizado/extracao do cranio/3_dif.png", dif)

    dif = dif * 255
    ret, thresh = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow(' 5 - Thresh diferença', thresh)
    cv2.imwrite("C:/Users/Lapisco02/Desktop/atualizado/extracao do cranio/4_thresh_dif.png", dif)

    #################################
    # Detecção da maior componente
    #################################
    img_norm_2 = np.zeros_like(img)

    #################################
    # Normalizacao da diferença
    #################################
    cv2.normalize(thresh, img_norm_2, 0, 255, cv2.NORM_MINMAX)
    img_norm_2 = cv2.convertScaleAbs(img_norm_2)

    #########################################################
    connectivity = 4
    output = cv2.connectedComponentsWithStats(img_norm_2, 4, cv2.CV_8U)
    labels = output[1]  # AM: Rotulo das componentes
    stats = output[2]  # AM: Estatistica das componentes
    centroids = output[3]  # AM: Centroids das componentes

    #########################################################

    img_max_ = np.zeros(img_norm_2.shape, img_norm_2.dtype)

    largecomponent1 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()

    stats[largecomponent1, cv2.CC_STAT_AREA] = largecomponent1

    img_max_[labels == largecomponent1] = 255
    img_max_[labels != largecomponent1] = 0

    cv2.imshow(' 6 - Maior componente do thresh diferenca', img_max_)
    cv2.imwrite("C:/Users/Lapisco02/Desktop/atualizado/extracao do cranio/5_maior_comp.png", img_max_)

    dif = (dif / 255)
    cv2.imshow('  - TESTE', dif)
    cv2.imwrite("C:/Users/Lapisco02/Desktop/atualizado/extracao do cranio/teste.png", dif)

    img_max_ = dif - img_max_
    cv2.imshow('  - TESTE - 2', img_max_)
    cv2.imwrite("C:/Users/Lapisco03/Desktop/atualizado/extracao do cranio/teste_2.png", img_max_)

    #################################
    # Normalizacao da diferença
    #################################
    img_norm_3 = np.zeros_like(img_max_)

    cv2.normalize(img_max_, img_norm_3, 0, 255, cv2.NORM_MINMAX)
    img_norm_3 = cv2.convertScaleAbs(img_norm_3)

    #################################
    # Clip Contrast
    #################################
    mean, std = cv2.meanStdDev(img_norm_3)  # Retirei por ultimo
    img_norm_3[img_norm_3 < mean] = 0

    cv2.imshow(' TESTE 3 imgnorm_3', img_norm_3)
    cv2.imwrite("C:/Users/Lapisco02/Desktop/atualizado/extracao do cranio/teste_3.png", img_norm_3)

    cv2.imshow(' TESTE 4 imgnorm_3/255', (img_norm_3 / 255))

    new_image = (dif - (img_norm_3))
    cv2.imshow(' 7 - Output image', new_image)
    cv2.imwrite("C:/Users/Lapisco03/Desktop/atualizado/extracao do cranio/cranio_88.png", new_image)

    cv2.waitKey()
    return new_image


#######################
# Principal
#######################
if __name__ == '__main__':
    print("""""")
    cases_name = []
    cont = 0
    img_path = "testimages"
    alpha = 1300
    num_iters = 800
    sigma = 5.5
    threshold = 0.34
    balloon = -1.0
    smoothing = 1

    for i in range(1, 2):
        fig = 'cranios/cranio_' + str(i) + '.png'
        cases_name.append(fig)

    cont = 1
    for case_name in cases_name:
        img_source = img_path + "/" + case_name
        img = imread(img_source)
        if img.ndim == 3:
            img = img[..., 0] / 255.0
        else:
            img = img / 255.0

        # img = new_image(img)
        test_GAC(cont, img, p_alpha=alpha, p_sigma=sigma, p_threshold=threshold, p_balloon=balloon,
                 p_smoothing=smoothing, p_num_iters=num_iters)
        cont = cont + 1
        ppl.close()
