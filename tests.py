# -*- coding: utf-8 -*-

from __future__ import division

import sys
import timeit

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append('roi')
sys.modules['Image'] = Image


def rgb2gray(img):
    """
    Image conversion from rgb to gray
    :param img:
    :return:
    """
    return 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


# ----------------------------------------------------
# AM: Criacao da funcao inicial do levelset
# Nesta funcao eh construida uma matrix binaria com o circulo de centro (center: cY,cX)
# com raio igual a sqradius, a regiao com 1's define o levelset 0.5
# ----------------------------------------------------
def circle_levelset(shape, center, sqradius, scalerow=1.0):
    # -------------------------------------------------------------------
    # AM: Constrói a função phi do level-set inicial baseado em to-do o
    # domínio da imagem (grid: WxH), calculando os pontos internos e externos
    # a função phi inicial
    # 1 - Cria uma malha com as mesmas dimensoes da imagem
    # 2 - Calcula a transposta de (1)
    # 3 - De (2) subtrai o ponto central(cx,cy) de cada coluna: c1-cx, c2-cy
    # 4 - Calcula a transposta do resultado de (3)
    # 5 - De (4) eleva todos os elemento ao quadrado
    # 6 - De (5) soma o elementos da primeira com a segunda matriz, resultando na Matriz Distancia
    # 7 - Calcula a raiz quadrada de (6)
    # 8 - Subtrai o raio (sqradius) de (7)
    # 9 - Define u como a regiao inicial (Matrix zero levelset) onde os pontos
    # > 0 em (8) estao dentro da levelset zero e os <= 0 estao fora. A matriz u eh
    # uma matrix binaria, 0 -> fora da regiao, 1 -> dentro da regiao
    # -------------------------------------------------------------------
    print("shape: ", shape)
    data = np.random.randint(5, size=(512, 256))

    grid = np.mgrid[list(map(slice, data.shape))].T - center
    phi = sqradius - np.sqrt(np.sum(grid.T ** 2, 0))

    u = np.float_(phi > 0)

    return np.hstack((u, u))


def test_gac(img, p_alpha=1000, p_sigma=4, p_threshold=0.45,
             p_balloon=1, p_smoothing=1, p_num_iters=70):
    from roi.startPointDoubleLung import get_initial_point_lung
    import morphsnakes

    # p_threshold = 0.50 #(Estabilidade prematura)

    #####################################################################
    # AM: Definicao do criterio de parada baseado no gradiente
    # das fronteiras/bordas do objeto com função de suavizacao gaussiana
    # Onde:
    # - o parametro alpha funciona como um fator de peso para o gradiente.
    # Quanto maior for este valor maior a influencia da magnitude do
    # gradiente em regioes de diferentes contrastes
    # - o parametro sigma atua definindo a regiao de influencia da funcao
    # gaussiana, ou seja, define o desvio padrao. Quanto maior este valor,
    # maior sera a regiao de suavizacao da magnitude do gradiente, suavizando
    # o gradiente em regioes onde ha diferentes contrastes.
    #####################################################################
    # g(I)
    levelset_zero, grad = get_initial_point_lung(img)
    g_i = morphsnakes.gborders(img, p_alpha, p_sigma)

    ee_3x3 = np.ones((4, 4), dtype=int)

    g_i = cv2.dilate(g_i, ee_3x3)

    g_i[g_i < 0.33] = 0.10
    g_i[0.30 < g_i.all() < 0.50] = 0.34
    ###################
    g_i = cv2.erode(g_i, ee_3x3)
    g_i = cv2.dilate(g_i, ee_3x3)

    # AM: Inicializacao do level-set em toda a dimensão da imagem
    # smoothing: scalar
    #       Corresponde ao numero de repeticoes em que a suavizacao sera
    # aplicada a cada iteracao. Ou seja, a cada passo, serao aplicadas
    # smoothing vezes a funcao de suavizacao. Este procedimento e realizado
    # na funcao step da classe MorphGAC. Este representa o parametro µ.
    #
    # threshold : scalar
    #     The threshold that determines which areas are affected
    #     by the morphological balloon. This is the parameter θ.
    # balloon : scalar
    #     The strength of the morphological balloon. This is the parameter ν.
    mgac = morphsnakes.MorphGAC(g_i, p_smoothing, p_threshold, p_balloon)

    # AM: Define a função phi inicial no domínio completo da imagem
    # (img.shape). Cujo centro é definido pelos pontos (iniPointY, iniPointX)
    # O raio da função phi inicial é definido último parametro, ex: 30.
    mgac.levelset = levelset_zero
    # AM: Visualiza a evolução da curva e inicializa o máximo de interações
    mgac.run(p_num_iters)
    return mgac.levelset


if __name__ == '__main__':
    from data_information import data_extraction

    # import matplotlib.pyplot as ppl

    img_path = 'datasets/ImagensTC_Pulmao_grayscale/'
    doc_binary_path = 'binary_images/doctor_images/'
    results_path = 'results/file_results/'

    alpha = 1500
    iterations = 100
    radius = 20
    auto_ini = False
    x_ini = 130
    y_ini = 250
    sigma = 4
    threshold = 0.35
    balloon = 1
    smoothing = 1

    ini_img = 20
    end_img = ini_img + 1

    img_paths_list = []
    for i in range(ini_img, end_img):
        img_paths_list.append(img_path + "{}.bmp".format(i))

    time = []
    df = pd.DataFrame()
    for image in img_paths_list:
        doc_img = cv2.imread(doc_binary_path + '%d.png' % ini_img, 0)

        print('img', ini_img)

        ini_img += 1

        dcm_img = cv2.imread(image, 0)

        dcm_img = dcm_img / 255.0

        start = timeit.default_timer()

        gac_result = test_gac(dcm_img, alpha, sigma, threshold,
                              balloon, smoothing, iterations)

        stop = timeit.default_timer()

        ee = np.ones((5, 5), dtype=int)
        gac_result = cv2.dilate(gac_result, ee)

        # ppl.imshow(gac_result)
        # ppl.imshow(doc_img)
        # ppl.show()
        # cv2.imshow('doc', doc_img * 255)
        # cv2.imshow('res', result * 255)
        # cv2.waitKey(0)

        measures = data_extraction.PerformMeasure(gac_result, doc_img)
        measures.print_measures()
        print('Time: ', stop - start)

        measures = data_extraction.PerformMeasure(gac_result, doc_img)
        pandas_result = pd.DataFrame(measures.return_measures()).T
        df = df.append(pandas_result, ignore_index=True)
        # print(' ')

    # df.index += 1
    # df.columns = [
    #     'accuracy',
    #     'matthews',
    #     'sensitivity',
    #     'dice',
    #     'hausdorff',
    #     'jaccard'
    # ]
    # df = df.drop(df.index[1])
    # df_mean = df.mean(axis=0)
    # df.to_csv(results_path + "pandas_metrics.csv")
    # df_mean.to_csv(results_path + "pandas_metrics_mean.csv")
