from scipy.ndimage.filters import generic_filter
from scipy.signal import convolve2d, medfilt2d
from scipy.spatial.distance import cdist
from os.path import abspath, join, dirname
import matplotlib.pyplot as ppl
from time import time
import numpy as np
import math
import sys
import cv2
import numexpr as ne

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from data_information import dcm_information as di


def mls_parzen(imgo, mask, dt, N, Lo, flag, shapes, a):
    """
    Esse metodo realiza uma analise das Faixas de densidade humanas
    - Mestrado em Engenharia de Telecomunicacoes
    - Elizangela Reboucas
    - 02/03/2016
    Level set proposed by Marques and Mederos in:
    a fast level set segmentation algorithm
    Submited to: Pattern Recognition Letters
    INPUT:
        imgo: image
        N: maximum number of iterations
        dt: delta t
    """

    stop2_div = False
    show = False

    x, y = shapes
    window = 9

    # if flag == 1:
    #     window = 5

    # [Matheus] Otimizacao do codigo para os mesmos valores nao sejam calculados mais de uma vez
    win_mut = window * window
    win_m_2 = - (2 * win_mut)
    lo_size_mul = x * y
    parzen_eq = 1 / ((window * math.sqrt(math.pi * 2)) ** 3)
    imgo_reshape_temp = np.zeros((x, x, 3))
    imgo_reshape_temp[:, :, 0] = imgo

    i = np.zeros((x, x, 3))
    i[:, :, 0] = np.copy(imgo)
    i[:, :, 1] = conv2(imgo, np.tile(1 / win_mut, (window, window)))
    i[:, :, 2] = generic_filter(imgo, np.std, window)
    i = i.reshape((lo_size_mul, 3))

    s_eg = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=np.uint8)
    f = np.zeros((x, y))

    int1 = np.uint8(1)
    int0 = np.uint8(0)
    # Inicialization of Surface and others
    psi=np.where(Lo>0,1,-1)
    psi=np.int16(psi)
    # psi = (2 * Lo) - 1  # Superficie inicial: cones
    psi_d = cv2.dilate(psi, s_eg)  # Primeira dilatacao
    psi_e = cv2.erode(psi, s_eg)  # Primeira erosao
    # Main procedure
    c = 1
    phi0 = np.ones((x, y))
    thr = np.ones((1, N))

    and_lamd = lambda ps, mas: imgo[np.where((ps == 1) & (mas == 1))]

    # Plot segmentation
    if show:
        fig = ppl.figure(figsize=(12, 12))
        ax3 = fig.add_subplot(1, 1, 1)
        ax3.imshow(imgo, cmap='gray')

    while np.sum(thr == 0) < 1:  # Convergence
        # print('sum thr == 0 < 10', np.sum(thr == 0) < 10 * 1)
        print('[INFO] C =', c)
        # time_while = time()
        c = c + 1
        psi = medfilt2d(psi + (dt * ((np.maximum(f, 0) * psi_d) - (np.minimum(f, 0) * psi_e))))
        psi_d = cv2.dilate(psi, s_eg)
        psi_e = cv2.erode(psi, s_eg)
        psi_bigger_1 = np.where(psi > 0, int1, int0)
        phi = bwperim(psi_bigger_1, shapes)
        # threshold of convergence (front stabilization)
        thr[0, c - 1] = np.sum(np.absolute(phi0 - phi))
        phi0 = np.copy(phi)
        # Parameter Estimation
        # print('time while', time() - time_while)
        if (c > 60 and c % 5 == 0) or c == 2:
            # time_if = time()
            and1 = and_lamd(psi_bigger_1, mask)
            and2 = and_lamd(np.where(psi < 0, int1, int0), mask)
            stroke_t = np.zeros((win_mut, 3))
            stroke_f = np.zeros((win_mut, 3))
            stroke_t[:, 0] = and1[np.random.permutation(and1.shape[0])[0:win_mut]]
            stroke_t[:, 1] = np.mean(and1)
            stroke_t[:, 2] = np.std(and1, ddof=1)
            stroke_f[:, 0] = and2[np.random.permutation(and2.shape[0])[0:win_mut]]
            stroke_f[:, 1] = np.mean(and2)
            stroke_f[:, 2] = np.std(and2, ddof=1)
            # print('time strokes', time() - time_if)
            # time_if = time()

            prob1 = pdf_parzen(stroke_t, i, win_m_2, x, parzen_eq)
            prob2 = pdf_parzen(stroke_f, i, win_m_2, x, parzen_eq)
            # print('time parzen total', time() - time_if)
            # time_if = time()

            f = prob1 - prob2
            f = (f * np.where(f > 0, 1, 0) / np.amax(f)) + \
                (f * np.where(f < 0, 1, 0) / - np.amin(f))
            f[mask == 0] = - 1
            # print('time if', time() - time_if)

        # Visualizar imagem
        if show and c % 1 == 0:
            ax3.contour(phi, [1], colors='r')
            fig.canvas.draw()
            ppl.pause(0.000000001)
            ppl.title('c = {}'.format(c))
            del ax3.collections[0]

    img = imgo * -phi

    # Alan
    aux = np.where(psi > 0, int1, int0)
    if np.sum(aux) == 0:
        stop2_div = True

    return phi, img, psi, stop2_div


def pdf_parzen(janela_classe, img, h, img_shape, parzen_eq):
    # time_for = time()
    # print('time parzen 2', time() - time_for)
    # p = np.zeros((img_shape * img_shape, 3))
    # for i in np.nditer(p):
    #     print(i)
    #     print(janela_classe[0, 0])
    #     # print(p[i])
    #     p[int(i)] = parzen_eq * pow(math.e, -(d[int(i)] / h))
    d = cdist(img, janela_classe, 'euclidean')
    time_for = time()
    p = np.mean(parzen_eq * ne.evaluate('exp(d / h)').T, axis=0)
    # print('time parzen 3', time() - time_for)
    return p.reshape((img_shape, img_shape))


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def bwperim(bw, shapes):
    rows = shapes[0] - 1
    idx = np.where(bw[1:, :] == bw, 1, 0) & \
          np.where(bw[:, 1:] == bw, 1, 0) & \
          np.where(bw[:rows, :] == bw, 1, 0) & \
          np.where(bw[:, :rows] == bw, 1, 0)
    return (1 - idx) * bw
