"""
Mestrado em Engenharia de Telecomunicacoes
Elizangela Reboucas
02/03/2016
"""
from os.path import abspath, join, dirname
from scipy.signal import convolve2d
import matplotlib.pyplot as ppl
from time import time
import numpy as np
import skimage
import sys
import cv2

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from data_information import dcm_information as di
from withinSkull import withinSkull
from mls_parzen import mls_parzen, conv2
import FolClustering as fl

resultTotalDensRLSParzen2 = []
tempTotalDensRLSParzen2 = []

debug = False
count_debug = 1
flag = 1

for r in range(1, 2):
    print("Repeticao", r)
    tempDensRLSParzen = []

    for z in range(1, 101):
        # Carregar a imagem Original
        imgOrig = di.load_dcm("../datasets/OriginalCT/Imagem{}.dcm".format(z))
        img_original = np.copy(imgOrig)
        atemp = np.multiply(2.5, np.subtract(np.double(np.double(imgOrig)), 1024))
        img = di.m_uint8(atemp)
        imgOrig = imgOrig - 1024

        imgOrig = cv2.resize(imgOrig, (256, 256))
        img = cv2.resize(img, (256, 256))
        img_original = cv2.resize(img_original, (256, 256))

        # [Matheus] Mediana na imagem DICOM original
        imgOrig = np.int16(cv2.medianBlur(imgOrig, 3))

        _, skullInside, skull, se, shapes = withinSkull(di.m_uint8(imgOrig),
                                                        flag)

        # imgUtil e uma imagem 16 bits somente da area ja erodida
        skullInside=np.where(skullInside>=1,1,0)
        skullInside=np.uint8(skullInside)
        imgUtil = cv2.medianBlur(imgOrig * skullInside, 3)

        # Imagens muito pequenas
        if 28 < z <= 31:
            flag = 1
        else:
            flag = 0

        # [Matheus] Calcula tempo em segundos
        # [Matheus] Cria mask de 11x11 e divide cada posicao por 81
        mask = np.tile(0.1111, (9, 9))
        var = np.where((43 < imgUtil) & (imgUtil < 76), 1, 0)  # 43 - 76
        tic = time()

        if 90 < z <= 96:
            _str = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]])
            var = cv2.erode(var.astype(np.int8), _str.astype(np.int8))
        elif z > 96:
            _str = np.array([[0, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1],
                             [0, 1, 1, 1, 0],
                             [0, 0, 1, 0, 0]])
            var = cv2.erode(var, _str)

        densidade = conv2(var, mask, 'same')
        int255 = np.uint8(255)
        int0 = np.uint8(0)
        imgPorcSang = np.where(densidade > 5, int255, int0)

        connectivity = 8
        output = cv2.connectedComponentsWithStats(imgPorcSang, connectivity, cv2.CV_8U)

        # Pegar as componentes do maior para menor
        # Sum of each component = output[2]
        indice1 = np.argsort(output[2][:, -1])[::-1]

        # [Matheus] The second biggest component is the AVC
        # Label matrix = output[1]
        avc_component = np.where(output[1] == indice1[1], 1, 0)
        imFinal = np.multiply(avc_component, imgUtil)

        if debug:
            count_debug = di.show_figures(imFinal, count_debug)
            count_debug = di.show_figures(avc_component, count_debug)
            # ppl.show()

        c = skimage.measure.regionprops(avc_component)[0].centroid

        xc = int(c[0] - 10)
        yc = int(c[1] - 10)
        imSaida = np.zeros(shapes)
        # [Matheus] Position where the centroid was misplaced
        imSaida[xc + 1: xc + 20, yc + 1: yc + 20] = se

        # Fast LEvel SEt Parzen
        X = skullInside * imgOrig
        mask = skullInside
        Lo = np.copy(imSaida)  # Rotulo==valorRot
        dt = 0.9
        N = 100
        shapes = img_original.shape

        orig_log = fl.clustering_log(img_original)

        phi, img, Psi, stop2_div = mls_parzen(X, mask, dt, N, Lo, flag, shapes, orig_log)
        temp = time()  # [Matheus] Para contagem Finish Time
        print('time parzen', temp - tic)
        output_final = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_8U)
        rotulo1 = output_final[1]  # Label matrix
        valorArea = output_final[2][:, -1]  # Sum of each component

        # Pegar as componentes do maior para menor
        indice1 = np.argsort(valorArea)[::-1]

        mask_comp = np.where(rotulo1 == indice1[1], 1, 0)

        pos = skimage.measure.regionprops(mask_comp)[0].centroid

        imgzero = np.zeros(X.shape, dtype=np.uint64)
        # imgzero = np.copy(Psi)
        # Psi = np.copy(imgzero)

        if debug:
            count_debug = di.show_figures(Psi, count_debug)
            count_debug = di.show_figures(imgzero, count_debug)
            ppl.show()

        cv2.imwrite("../datasets/results/Imagem{}.png".format(z), phi*255)
        # if Psi.any() > 0:
        #     # [Matheus] Codigo para representar 'floodfill' do MATLAB
        #     # im_flood_fill = np.copy(Psi).astype(np.uint64)
        #     # # Mask used to flood filling.
        #     # # NOTE: the size needs to be 2 pixels bigger on each side
        #     # # than the input image
        #     # h, w = Psi.shape[:2]
        #     # mask = np.zeros((h + 2, w + 2), np.uint64)
        #     # # Floodfill from point (0, 0)
        #     # cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
        #     # # Invert floodfilled image
        #     # # Combine the two images to get the foreground
        #     # inside = np.bitwise_or(Psi, cv2.bitwise_not(im_flood_fill))
        #     # Psi = np.where(inside > 0, 255, 0)
        #     cv2.imwrite("../datasets/results/Imagem{}.png".format(z), Psi)
        # else:
        #     cv2.imwrite(
        #         "../datasets/results/Imagem{}.png".format(z), phi*255
        #     )

        # tempDensRLSParzen.append(temp)

        # resultDensRlSParzen = rumTestResult('ResultsDensRLSParzen')

        # resultTotalDensRLSParzen2 = [resultTotalDensRLSParzen2; resultDensRLSParzen];
        # tempTotalDensRLSParzen2 = [tempTotalDensRLSParzen2; tempDensRLSParzen];

    # tabela_resultado('Tabela de Resultados DensRLSParzen - Total', 'DensRLSParzen',
    # resultTotalDensRLSParzen2, tempTotalDensRLSParzen2);

    # save('resultTotalDensRLSParzen2.mat', 'resultTotalDensRLSParzen2');
    # save('tempTotalDensRLSParzen2.mat', 'tempTotalDensRLSParzen2');
