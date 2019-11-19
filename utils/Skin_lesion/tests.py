import morphsnakes

import numpy as np
from matplotlib import pyplot as plt
import time

import sys
from automaticInitialization import SetInitialPoint, choose_channel, mean_pixels, normalize_image
import cv2

import glob
import os
from ParzenWindow import ParzenWindow

images = []


def test_GAC(original, img, p_alpha=1000, p_auto_ini=True, p_x_ini=0, p_y_ini=0, p_sigma=6, p_threshold=0.47,
             p_balloon=1, p_smoothing=1, p_num_iters=70, p_raio=30, p_count=0):

    ############## Fazer função em outra biblioteca para isso #########################
    original_resized = cv2.resize(original, (0, 0), fx=0.2, fy=0.2)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

    processedImage = img
    
    #cv2.imshow("resized_image", processedImage)
    #cv2.waitKey(0)


    #if (len(img.shape) <= 2):
    #processedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #cv2.imshow("Histograma", hist)
    #cv2.waitKey(0)

    #processedImage = cv2.equalizeHist(processedImage)
    '''
    cv2.imshow("Imagem equalizada", processedImage)
    cv2.waitKey(1)
    '''
    #else:
        #processedImage = img
    
    for i in range(1):
        processedImage = cv2.GaussianBlur(processedImage, (3, 3), 0)
        processedImage = cv2.medianBlur(processedImage, 3)
        #cv2.imshow('Blur', processedImage)
        #cv2.waitKey(0)

    ret, binaryImage = cv2.threshold(processedImage,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #cv2.imshow('Threshold', processedImage)
    #cv2.waitKey(0)
    
    #==================== Mask =================
    mask_img = cv2.imread('PH2Dataset/mask.png', 0)

    mask_img = cv2.resize(mask_img, (0, 0), fx=0.4, fy=0.4)
    mask_img = cv2.resize(mask_img, (binaryImage.shape[1], binaryImage.shape[0]))

    binaryImage = mask_img & binaryImage

    #cv2.imshow('Result', result)
    #cv2.waitKey(0)

    #============================================

    kernel = np.ones((5, 5), np.uint8)
    binaryImage = cv2.dilate(binaryImage, kernel, iterations=1)
    
    #cv2.imshow('Dilate', processedImage)
    #v2.waitKey(1)

    # Fazer uma função que determine os limites de threshold individualmente para cada imagem (medidas estatísticas) 

    #return

    ####################################################################################
 
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
    #gI = morphsnakes.gborders(img, alpha=600, sigma=3)
    #gI = morphsnakes.gborders(processedImage, alpha=p_alpha, sigma=p_sigma)

    gI = morphsnakes.gborders2(binaryImage)


    #plt.title("Resultado da função gI")
    #plt.imshow(gI)    
    #return
    
    #####################################################################
    # AM: Inicializacao do level-set em toda a dimensão da imagem
    # smoothing: scalar
    #       Corresponde ao numero de repeticoes em que a suavizacao sera
    # aplicada a cada iteracao. Ou seja, a cada passo, serao aplicadas
    # smoothing vezes a funcao de suavizacao. Este procedimento e realizado 
    # na funcao step da classe MorphGAC. Este representa o parametro µ.    
    #
    # threshold : scalar
    #     Determina o limiar de atuacao da forca balao. Eh o parametro  θ.
    # balloon : scalar
    #     Determina a intensidade/velocidade da forca balao. Eh o parametro ν.
    ##################################################################

    #AM: mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
    mgac = morphsnakes.MorphGAC(gI, smoothing=p_smoothing, threshold=p_threshold, balloon=p_balloon)    

    ##################################################################
    # AM: Calcula o ponto de inicialização da curva inicial do level-set
    ##################################################################

    #mgac.levelset = SetInitialPoint(img)

    img = img & mask_img

    mean = int(np.sum(img) / np.alen(img))

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row, col] == 0:
                img[row, col] = abs((img[row, col] - mean) / 2)

    parzen_window = ParzenWindow(img, inf_thresh=0, sup_thresh=100, h=0.6, n_points=50)

    mgac.levelset = parzen_window.segmentation()
    cv2.imshow('Original', img)

    data = mgac.levelset
    data = data / data.max()
    data = data * 255
    images.append(~data.astype(np.uint8))

    cv2.imshow('Segmentation', ~data.astype(np.uint8))
    cv2.waitKey(0)

    ##################################################################
    # AM: Visualiza a evolução da curva e inicializa o máximo de interações
    ##################################################################


    #plt.figure()
    #morphsnakes.evolve_visual(mgac, original.shape[0], original.shape[1], num_iters=p_num_iters, background=original_resized, count = p_count)
    
    #save_name = 'D:/Artigos/PH2Dataset/Results/IMD'+ str(p_count) + '.png'
    #plt.savefig(save_name)
    #plt.close()


#######################
# Principal
#######################
if __name__ == '__main__':
    start_time = time.time()

    archive = open('Performance_Results.txt', 'w')

    img_path = 'PH2Dataset/Exam'

    cases_name = glob.glob(os.path.join(img_path, '*.bmp'))

    img_sample = cv2.imread(cases_name[36])

    blue = choose_channel(img_sample, 0)
    mean_1 = mean_pixels(img_sample)

    alpha = 1000
    num_iters = 25
    auto_ini = True
    sigma = 1
    threshold = 0.1
    balloon = 1
    smoothing = 1
    
    actual_time = float(0)
    sum_time = 0

    for i, case_name in enumerate(cases_name):
        original_img = cv2.imread(case_name)

        img = choose_channel(original_img, 0)

        img = normalize_image(img, mean_1)

        test_GAC(original_img, img, p_alpha=alpha, p_auto_ini=auto_ini, p_sigma = sigma, p_threshold=threshold,
                 p_balloon=balloon, p_smoothing=smoothing, p_num_iters=num_iters, p_count=i)

        plt.show()
        image_time = (time.time() - start_time) - actual_time 

        sum_time += image_time

        tempo = 'Image ' + str(i) + ': ' + str(image_time) + 's\n'

        archive.write(tempo)

        actual_time = time.time() - start_time
        print("--- {} seconds ---".format(image_time))

    meanTime = sum_time/len(cases_name)

    tempo = 'Media: ' + str(meanTime) + 's'

    archive.write(tempo)
    archive.close()

    print('[INFO] Saving Images...')

    for i, image in enumerate(images):
        if not os.path.exists('output'):
            print('[INFO] Creating output path...')
            os.makedirs('output')

        cv2.imwrite('output/Save {}.png'.format(i), image)
