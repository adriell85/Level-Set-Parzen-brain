from scipy.ndimage.filters import generic_filter
from os.path import abspath, join, dirname
from scipy.spatial.distance import cdist
from scipy.signal import convolve2d
from scipy.signal import medfilt2d
from time import time
from matplotlib import pyplot as ppl
import numpy as np
import math
import sys
import cv2
from data_information import dcm_information as di
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from bwperin import bwperin


def mls_parzen(imgo, mask, dt, N, M, Lo, X1, flag, count_debug):
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
    #x e y pegam os valores do tamanho da imagem do crânio
    x, y = imgo.shape

    window = 13

    if flag == 1:
        window = 5

    # [Matheus] Otimizacao do codigo para os mesmos valores nao sejam calculados mais de uma vez
    win_mut = window * window
    win_m_2 = (2 * win_mut)
    lo_size_mul = x * y
    parzen_eq = 1 / ((window * math.sqrt(math.pi * 2)) ** 3)
    imgo_reshape_temp = np.zeros((x, x, 3))
    imgo_reshape_temp[:, :, 0] = imgo
    #cria matriz redimensionada(quadrada), com coloração vermelha, na qual a região em vermelho é a região da massa
    #cinzenta do cerebro
    #count_debug = di.show_figures(imgo_reshape_temp, count_debug)
    #ppl.show()

    i = np.zeros((512, 512, 3))
    i[:, :, 0] = np.copy(imgo)
    #cria matriz i repassando valores de imgo(pintada a região da massa cinzenta em vermelho)
    #count_debug = di.show_figures(i, count_debug)
    #ppl.show()
    i[:, :, 1] = conv2(imgo, np.tile(1 / win_mut, (window, window)))
    #recolore a matriz i, na qual a região vermelha deixa de ser vermelha e passa a ser amarela com as bordas verdes
    #count_debug = di.show_figures(i, count_debug)
    #ppl.show()
    i[:, :, 2] = generic_filter(imgo, np.std, window)
    #esta linha retira a coloração amarela, deixando apenas a região verde, que por conseguinte se torna azul
    #np.std: calcula o desvio padrão
    #count_debug = di.show_figures(i, count_debug)
    #ppl.show()
    i = i.reshape((lo_size_mul, 3))
    #modifica as imenções de i, para um vetor


    s_eg = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=np.uint8)
    f = np.zeros((x, y))

    # Inicialization of Surface and others
    int1 = np.uint8(1)
    int0 = np.uint8(0)

    psi = (2 * Lo) - 1  # Superficie inicial: cones
    #count_debug = di.show_figures(psi, count_debug)
    #ppl.show()
    psi_d = cv2.dilate(psi, s_eg)  # Primeira dilatacao
    psi_e = cv2.erode(psi, s_eg)  # Primeira erosao
    #abre a imagem, dilata e erode uma vez(pequenas)

    # Main procedure
    c = 1
    phi0 = np.ones((x, y))
    thr = np.ones((1, N))
    #print(thr)

#----------------------------------------------------------------------------------------------------------------------
    #Plot segmentation
     #parte responsável pelo crescimento da região vermelha por cima da imagem original
    fig = ppl.figure(figsize=(12, 12))
    ax3 = fig.add_subplot(1, 1, 1)
    ax3.imshow(imgo, cmap='gray')
    #plotando a imagem responsável por mostrar o processo de segmentação(crescimento da região vermelha)


#parte do código responsável por crescer a imagem de entrada até que esta atinja a imagem de saída

    while (c < N) * 1 and (np.sum(thr == 0) < 10) * 1:  # Convergence
        # A condicional se resume a, primeiro, o procedimento não pode exceder o número máximo de interações que é 400
        # já na condicional depois do and: a imagem final( borda da região de interesse em branco), é subtraida da
        # segmentação(procedimento ocorrendo a cada interação), depois de somadas, todos os píxels são somados em um
        # único elemento e armazenados na matriz thr, cada termo é comparado a 0.
        #  se as imagens forem iguais(segmentação completa), o resultado armazenado no vetor thr será zero, logo é a forma
        # dada pela condicional do while para saber se o procedimento de segmentação baseado na janela de parzem estará concluido
        # este vetor thr deverá ser perguntado 10 vezes, se por 10 vezes o ponto do vetor thr for 0, o código fecha o while
        print('sum thr == 0 < 10', np.sum(thr == 0) < 10 * 1)
        time_while = time()
        c = c + 1
        # contador de interações
        # imagem do  ponto(centroide)
        psi = medfilt2d(psi + (dt * ((np.maximum(f, 0) * psi_d) -(np.minimum(f, 0) * psi_e))))
        #count_debug = di.show_figures(i, count_debug)
        #ppl.show()

        # dilata e erode o centroide
        psi_d = cv2.dilate(psi, s_eg)
        psi_e = cv2.erode(psi, s_eg)


        phi_1 = bwperin(np.where(psi > 0, int1, int0))
        #captura o perimetro dom centroide, como um canny
        #count_debug = di.show_figures(phi_1, count_debug)
        #ppl.show()

        phi = np.copy(phi_1)

        # threshold of convergence (front stabilization)
        thr[0, c - 1] = np.sum(np.absolute(phi0 - phi))
        #np.absolute:

        #count_debug = di.show_figures(phi_1, count_debug)
        #ppl.show()
        phi0 = np.copy(phi)
        #count_debug = di.show_figures(phi, count_debug)
        #ppl.show()

        # Parameter Estimation
        print('[INFO] C =', c)
        print('time while', time() - time_while)
        if (c > 60 or c == 2) and (c % 5 == 0 or c == 2):#condicionamento de execução
            #será executada em interações específicas
            time_if = time()
            #psi_bigger_i: imagem binarizada do centroide na qual o centroide é branco e o resto é preto
            psi_bigger_1 = np.where(psi > 0, int1, int0)
            #count_debug = di.show_figures(psi_bigger_1, count_debug)
            #ppl.show()
            #psi_lees_1:imagem binarizada do centroide,na qual o centroide é preto e o resto é branco
            psi_less_1 = np.where(psi < 0, int1, int0)
            #count_debug = di.show_figures(psi_less_1, count_debug)
            #ppl.show()

            and1 = imgo[(psi_bigger_1 == 1) & (mask == 1)]

            and2 = imgo[(psi_less_1 == 1) & (mask == 1)]

            stroke_t = np.zeros((win_mut, 3))
            stroke_f = np.zeros((win_mut, 3))

            stroke_t[:, 0] = and1[np.random.permutation(and1.shape[0])[0:win_mut]]

            stroke_t[:, 1] = np.mean(and1)
            stroke_t[:, 2] = np.std(and1, ddof=1)

            stroke_f[:, 0] = and2[np.random.permutation(and2.shape[0])[0:win_mut]]
            stroke_f[:, 1] = np.mean(and2)
            stroke_f[:, 2] = np.std(and2, ddof=1)
            print('time strokes', time() - time_if)
            time_if = time()

            prob1 = pdf_parzen(stroke_t, i, win_m_2, x, parzen_eq)
            prob2 = pdf_parzen(stroke_f, i, win_m_2, x, parzen_eq)
            print('time parzen total', time() - time_if)
            time_if = time()

            f = prob1 - prob2
            f = (f * np.where(f > 0, 1, 0) / np.amax(f)) + \
                (f * np.where(f < 0, 1, 0) / - np.amin(f))
            f[mask == 0] = - 1

            print('time if', time() - time_if)

        # Visualizar imagem
        if c % 5 == 0:
            #plota imagem,com borda em cima
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
    time_for = time()
    d = cdist(img, janela_classe, 'euclidean')
    print('time parzen 2', time() - time_for)
    time_for = time()

    p = np.mean(parzen_eq * np.exp(-d / h).T, axis=0)
    print('time parzen 3', time() - time_for)
    return p.reshape((img_shape, img_shape))


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)
