"""
Mestrado em Engenharia de Telecomunicacoes
Elizangela Reboucas
02/03/2016
"""

from os.path import abspath, join, dirname
from scipy.signal import convolve2d
import matplotlib.pyplot as ppl
import time
import numpy as np
import skimage
import sys
import cv2
import math

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from data_information import dcm_information as di
from skimage.segmentation import clear_border
# from withinSkull import withinSkull
from mls_parzen import mls_parzen, conv2
# import FolClustering as fl
# from AreaLabel import AreaLabel
from cv2 import connectedComponentsWithStats, CV_8U
import Adriell_defs as ad
import pywt


resultTotalDensRLSParzen2 = []
tempTotalDensRLSParzen2 = []

debug = False
count_debug = 1
flag = 1

for r in range(1, 2):
    print("Repeticao", r)
    tempDensRLSParzen = []
    time_total = []

    for z in range(1, 101):
        # Carregar a imagem Original
        imgOrig = di.load_dcm("../datasets/OriginalCT/Imagem{}.dcm".format(z))
        # img_GT = cv2.imread("../datasets/resultados_GT/Imagem{}.png".format(z))
        img_original = np.copy(imgOrig)
        img_original= cv2.resize(img_original, (256, 256))
        # img_original,var2,var3,var4=ad.haar(img_original)
        # img_original=np.uint8(img_original)

        # img_GT = cv2.resize(img_GT, (256, 256))
        # TODO - Calcular o inicio do tempo aqui
        start = time.time()
        img_res= (img_original-1024)
        img_oss,img_oss_ant=ad.within_skull(img_res)

        img_oss,img_oss_erode=ad.image_morfologic1(img_oss_ant,1,2)
        # ==============================================================================================================


        M = cv2.moments(img_oss)

        cY = int(M["m10"] / M["m00"])
        cX = int(M["m01"] / M["m00"])
        img_oss_center=(cX,cY)
        img_oss[cX,cY]=0





        img_si,img_si_ant,img_si_ant_show= ad.mult(img_res,img_oss,img_oss_erode)


        # =========================================================================================================
        img_si_ant,img_si_ant_median= ad.thresolded_img(img_si_ant)

# =====================================================================================================================

        img_si_ant,img_si_after_erode = ad.image_morfologic2(img_si_ant)



        # =============================================================================================================

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(img_si_ant)
        # =============================================================================================================

        img_filtered= ad.centroid_operations1(centroids,img_oss_center,stats,n,labels)

        # =============================================================================================================
        M = cv2.moments(img_filtered)
        cY = int(M["m10"] / M["m00"])
        cX = int(M["m01"] / M["m00"])
        img_filtered_center = (cX, cY)

        # ===========================================================================================================================

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(img_si_after_erode)

        img_final=ad.centroid_operations2(centroids,img_filtered_center,labels,n)















        X = np.asarray(img_si_ant_show, np.int16)

        # count_debug = di.show_figures(img_sim, count_debug)
        # ppl.show()
        mask = img_oss

        img_final=cv2.medianBlur(img_final,3)

        Lo = (img_final)  # Rotulo==valorRot
        dt = 0.9
        N = 100
        shapes = img_original.shape






        orig_log = np.where(img_final > 0, 1, -1)
        orig_log = np.int16(orig_log)

        # count_debug = di.show_figures(orig_log, count_debug)
        # ppl.show()
        # cv2.imshow('img',img_filtered)
        # cv2.imshow('img_gt',img_GT)
        # cv2.imshow('img_si_ant_show',img_si_ant_show)

        # cv2.waitKey(0)




        phi, img, Psi, stop2_div = mls_parzen(np.asarray(X, np.uint8), mask, dt, N, Lo, flag, shapes,orig_log)
        ppl.close()
        end=time.time()
        print('time {0}'.format(end-start))
        time_total.append(end - start)


        cv2.imwrite("../datasets/resultsA/Imagem{}.png".format(z), phi * 255)
        print(z)
    print('average time:{0:.4f} +/- {1:.4f}'.format(np.mean(time_total),np.std(time_total)))
    file = open('../datasets/resultsA/results.txt', 'w')
    file.write('\naverage time:{0:.4f} +/- {1:.4f}'.format(np.mean(time_total),np.std(time_total)))
    file.close()

        # depois de adicionar jacc:
        # dice: 0.9731007575187614
        # sens: 0.9814008367945688
        # spec: 0.9842416887572993
        # accu: 0.9834408379004095
        # f1sc: 0.9726844028256346
        # matt: 0.977449336885461
        # jacc: 0.981686517748409

        # dice: 0.9211011275744342
        # sens: 0.9339255576425309
        # spec: 0.9991136186942999
        # accu: 0.9982867431640625
        # f1sc: 0.9211011275744342
        # matt: 0.9227965115677673
        # jacc: 0.9982867431640625
