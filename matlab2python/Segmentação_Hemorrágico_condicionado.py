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
from withinSkull import withinSkull
from mls_parzen import mls_parzen, conv2
import FolClustering as fl
from AreaLabel import AreaLabel
from cv2 import connectedComponentsWithStats, CV_8U

resultTotalDensRLSParzen2 = []
tempTotalDensRLSParzen2 = []

debug = False
count_debug = 1
flag = 1

for r in range(1, 2):
    print("Repeticao", r)
    tempDensRLSParzen = []
    time_total = []

    for z in range(90, 101):
        # Carregar a imagem Original
        imgOrig = di.load_dcm("../datasets/OriginalCT/Imagem{}.dcm".format(z))
        img_GT = cv2.imread("C:/Users/gomes/Documents/fgac-parzen-system-master/datasets/resultados_GT/Imagem{}.png".format(z))
        img_original = np.copy(imgOrig)
        img_original= cv2.resize(img_original, (256, 256))
        img_GT = cv2.resize(img_GT, (256, 256))
        # TODO - Calcular o inicio do tempo aqui
        start = time.time()
        img_res= (img_original-1024)
        # img_res=np.int16(cv2.medianBlur(img_res,5))
        # img_res[img_res<0]=0
        # var1,var2=AreaLabel(img_res)
        img_oss=np.where(img_res>100,0,1)
        img_oss=np.uint8(img_oss)
        img_oss_ant=np.where(img_oss<1,1,0)
        img_oss = np.uint8(img_oss)
        img_oss_ant=np.uint8(img_oss_ant)
        img_oss_ant = np.where(img_oss_ant < 1, 1, 0)
        img_oss_ant = np.uint8(img_oss_ant)

        kernel1 = np.asarray([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]],np.uint8)
        kernel2 =np.asarray([[0,1,1,1,1,1,0],
                             [1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1],
                             [0,1,1,1,1,1,0]], np.uint8)
        img_oss_erode=img_oss_ant

        if (z>=90)&(z<95):
          img_oss_erode = cv2.erode(img_oss_ant, kernel1,iterations=2)





        img_oss=clear_border(img_oss_erode)


        M = cv2.moments(img_oss)

        cY = int(M["m10"] / M["m00"])
        cX = int(M["m01"] / M["m00"])
        img_oss_center=(cX,cY)
        # img_oss[cX,cY]=0





        img_si = img_res * img_oss
        img_si_ant = np.copy(img_si)
        img_si_ant_show = np.copy(img_si)


        img_si = img_si * img_oss_erode



        img_si_ant=np.uint8(img_si_ant)
        img_si_ant = cv2.equalizeHist(img_si_ant)
        if z==28:
            ret, img_si_ant = cv2.threshold(img_si_ant, 220, 255, type=cv2.THRESH_TOZERO)
        if (z >=26)&(z<28):
            ret, img_si_ant = cv2.threshold(img_si_ant, 200, 255, type=cv2.THRESH_TOZERO)
        if (z >=90):
            ret, img_si_ant = cv2.threshold(img_si_ant, 180, 255, type=cv2.THRESH_TOZERO)
        else:
            ret, img_si_ant = cv2.threshold(img_si_ant, 213, 255, type=cv2.THRESH_TOZERO)
        cv2.imshow('normal2', img_si_ant)
        # parei aqui: em vez de umsar median, usar uma abertura
        img_si_ant_median=img_si_ant
        img_si_ant = cv2.medianBlur(img_si_ant, 3)



        # img_si=cv2.medianBlur(img_si,7)
        img_si_after_erode=img_si_ant
        img_si_ant = cv2.erode(img_si_ant, kernel2)
        img_si_ant = cv2.dilate(img_si_ant, kernel2)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(img_si_ant)
        from scipy.spatial import distance
        distances=[0]*n




        for i in range(n):
            a=(centroids[i,1],centroids[i,0])

            distances[i]= distance.euclidean(img_oss_center,a)




        #
        # # Loop through areas in order of size
        areas = [s[4] for s in stats]
        sorted_idx = np.argsort(np.unique(areas))
        areas_org=np.sort(areas)

        for k in range(n):
            if areas[k] < 60:
                    distances[k] = 1000000
        distances_org = np.sort(distances)

















        distance1=distances_org[1]

        for i in range(1,n):
            if distances[i]==distance1:
                val1=i

        img_filtered=np.where((labels==val1),255,0)
        img_filtered=np.uint8(img_filtered)







        # cv2.imshow('normal1',img_si)
        # cv2.imshow('normal', img_final*255)
        # cv2.imshow('normal2', img_si_ant)
        # cv2.waitKey(0)

        # X = img_original * np.asarray(img_oss_erode, np.int16)
        X = np.asarray(img_si_ant_show, np.int16)

        # count_debug = di.show_figures(img_sim, count_debug)
        # ppl.show()
        mask = img_oss
        Lo = np.copy(img_filtered)  # Rotulo==valorRot
        dt = 0.9
        N = 100
        shapes = img_original.shape




        # ==============================================================================================================

        # orig_log = fl.clustering_log(img_original)
        kernel3 = np.asarray([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]],np.uint8)


        orig_log = cv2.erode(img_filtered,kernel1)
        # orig_log = cv2.resize(orig_log, (512, 512))
        orig_log=np.where(img_filtered>0,1,-1)
        orig_log=np.int16(orig_log)
        # count_debug = di.show_figures(orig_log, count_debug)
        # ppl.show()
        cv2.imshow('img',img_filtered)
        cv2.imshow('img_gt',img_GT)
        cv2.imshow('img_si_ant_show',img_si_ant_show)

        cv2.waitKey(0)


        #
        #
        # phi, img, Psi, stop2_div = mls_parzen(np.asarray(X, np.uint8), mask, dt, N, Lo, flag, shapes,orig_log)
        # # TODO - Terminar de calcular o tempo aqui
        # end=time.time()
        # print('time {0}'.format(end-start))
        # # time_t= end-start
        # time_total.append(end - start)
        # cv2.imwrite("../datasets/resultsA/Imagem{}.png".format(z), phi * 255)
        print(z)
    # print('average time', np.mean(time_total)
          #este codigo será modificado de forma a tentar se obter melhores metricas

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
