import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_gradient_magnitude

def getInitialPointLung(img):
    
    cv2.imshow('Original', img)
    img_norm = np.zeros_like(img)

    #img_b, img_g, img_r = cv2.split(img)        #FX

    #################################
    # 1 - Normalizacao
    #################################
    #cv2.normalize(img_b, img_norm, 0, 255, cv2.NORM_MINMAX)     #FX
    
    cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
    img_norm = cv2.convertScaleAbs(img_norm)
        
    #################################    
    # 2 - Clip Contrast
    #################################
    mean, std = cv2.meanStdDev(img_norm)
    img_norm[img_norm < mean] = 0;
    
    cv2.imshow("imgmedia", img_norm)

    #################################
    # 3 - Binarizacao com Otsu
    #################################
    ret, img_bin = cv2.threshold(img_norm,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #################################
    # 4 - Filtragem com erosão da abertura morfologica
    #################################    
    ee = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]);
    # Abertura Morfológica
    imgerode = cv2.erode(img_bin, ee);
    #imgdilate = cv2.dilate(imgerode, ee);    
    # Erosão da abertura
    #imgerode = cv2.erode(imgdilate, ee);
    # Dilatacão com um E.E 
    ee2 = np.array([[1, 1 ],[1, 1], [1, 1], [1, 1], [1, 1]]); # FIXME: Talvez ajustar este E.E junte a componente do caso 9
    #imgdilate = cv2.dilate(imgerode, ee2);
    imgerode = cv2.dilate(imgerode, ee2);

    cv2.imshow('4 - Pos filtro img_bin', imgerode)
    #################################
    # 5 - Detecção da maior componente
    #################################    
    connectivity = 4  
    #output = cv2.connectedComponentsWithStats(img_bin, connectivity, cv2.CV_8U)    
    output = cv2.connectedComponentsWithStats(imgerode, connectivity, cv2.CV_8U)    
    labels = output[1]      # AM: Rotulo das componentes 
    stats = output[2]       # AM: Estatistica das componentes
    centroids = output[3]   # AM: Centroids das componentes

    img_max_ = np.zeros(img_bin.shape, img_bin.dtype)
    largecomponent1 = 1+stats[1:, cv2.CC_STAT_AREA].argmax()
    img_max_[labels == largecomponent1] = 255
    img_max_[labels != largecomponent1] = 0

    cv2.imshow("5 - Filtragem Maior Comp.", img_max_);
    #cv2.imshow("5 - Filtragem Maior Comp. - Bin", img_bin);

    #################################
    # 6 - Definição do perimetro baseado no centroide
    #################################    
    # identifica um centroid em img_max_  
    ray = 110  
    roi_cx = int(centroids[largecomponent1,0])
    roi_cy = int(centroids[largecomponent1,1])
    img_roi = np.zeros_like(img)    
    img_max_inverted = 255 - img_max_;
    # Separacao de componentes ligadas, isto evita erro na reconstrucao
    ee2 = np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    img_max_inverted = cv2.erode(img_max_inverted, ee2)
    ee2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    img_max_inverted = cv2.erode(img_max_inverted, ee2)

    #img_max_inverted = cv2.dilate(img_max_inverted, ee2)
    img_roicrop_rect = img_max_inverted[roi_cy-100:roi_cy+100, roi_cx-2*ray:roi_cx+2*ray]

    # corta uma ROI com centro no entroid em img_max_    
    cv2.imshow("6 - Definicao do Perimetro", img_roicrop_rect);    
    
    #################################
    # 7 - Identificação das duas maiores componentes
    #################################    
    # Identificar as duas maiores componentes    
    connectivity = 4  
    output = cv2.connectedComponentsWithStats(img_roicrop_rect, connectivity, cv2.CV_8U)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    
    img_max2_ = np.zeros(img_roicrop_rect.shape, img_roicrop_rect.dtype)
    largecomponent1 = 1+stats[1:, cv2.CC_STAT_AREA].argmax()
    stats[largecomponent1, cv2.CC_STAT_AREA] = largecomponent1
    largecomponent2 = 1+stats[1:, cv2.CC_STAT_AREA].argmax()

    # AM: Identifica, com branco, as componentes
    img_max2_[labels == largecomponent1] = 255  
    img_max2_[labels == largecomponent2] = 255          

    img_max_[:,:] = 0
    img_max_[roi_cy-100:roi_cy+100, roi_cx-2*ray:roi_cx+2*ray] = img_max2_
    cv2.imshow("7 - Identificacao das duas componentes", img_max_);
        
    #################################
    # 8 - Reconstrução morfológica das componentes
    #################################        
    img_max_ = img_max_ / 255
    img_max_inverted = img_max_inverted / 255
        
    diff = np.zeros_like(img_max_inverted)
    k = 200 #FIXME: Além do k maximo, tentar definir ponto de parada quele quando n houver mudancas
    index = 0
    #plt.show()
    ee = np.array([[1, 1, 1],[1, 1, 1], [1, 1, 1]]);
    while index < k:
        img_max_ = cv2.dilate(img_max_, ee)
        cv2.multiply(img_max_, img_max_inverted, img_max_)
        index = index + 1
        
    img_max_ = img_max_*255
    #cv2.imshow("8 - Recontrucao Marker", img_max_);
    cv2.imshow("8 - Recontrucao Mask", img_max_inverted);
    #ee = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
    #img_max_ = cv2.erode(img_max_, ee)
    cv2.imshow("8 - Recontrucao Erode", img_max_);
    
    #cv2.waitKey(0)

    return img_max_