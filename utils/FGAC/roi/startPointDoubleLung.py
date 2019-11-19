import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_gradient_magnitude
import operator

debug = False;
def getInitialPointLung(img):
    
    if debug:
        cv2.imshow('Original', img)
        img
        cv2.imwrite('roi/inicialization/1_original.png', img)
        
    img_norm = np.zeros_like(img)    
    #################################
    # 1 - Normalizacao
    #################################
    cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
    img_norm = cv2.convertScaleAbs(img_norm)
        
    #################################    
    # 2 - Clip Contrast
    #################################
    mean, std = cv2.meanStdDev(img_norm)
    ee = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    #img_norm = cv2.dilate(img_norm, ee)    
    #img_norm = cv2.medianBlur(img_norm,3)
    #img_norm = cv2.bilateralFilter(img_norm,9,155,155)
       
    if debug:
        cv2.imshow("imgmedia 1", img_norm)
        cv2.imwrite('roi/inicialization/2_img_norm.png', img_norm)
    img_norm[img_norm < mean*1.00] = 0;  
    if debug:  
        cv2.imshow("imgmedia 2- clip contrast", img_norm)
        cv2.imwrite('roi/inicialization/3_clip_constrast.png', img_norm) 
    img_norm = cv2.erode(img_norm, ee)    

    #img_norm = cv2.dilate(img_norm, ee)
    #img_norm = cv2.erode(img_norm, ee)
    
    img_norm = cv2.dilate(img_norm, ee)
    if debug:
        cv2.imshow("imgmedia 2- clip contrast abertura", img_norm)
        cv2.imwrite('roi/inicialization/3_clip_constrast_opening.png', img_norm) 
        
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

    if debug:
        cv2.imshow('3 - Binarizacao com Otsu', img_bin)
        cv2.imwrite('roi/inicialization/4_binarization_otsu.png', img_bin)
        cv2.imshow('4 - Pos filtro img_bin', imgerode)
        cv2.imwrite('roi/inicialization/5_filter_erode_dilate.png', imgerode)
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

    ee = np.array([[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    ee2 = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    img_max_ = cv2.dilate(img_max_, ee)
    img_max_ = cv2.dilate(img_max_, ee2)

    if debug:
        cv2.imshow("5 - Filtragem Maior Comp.", img_max_);
        cv2.imwrite('roi/inicialization/6_largest_component.png', img_max_)
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

    
    
    #AM: FIXME! Raio alterado para extrair imagem para o artigo
    #img_roicrop_rect = img_max_inverted[roi_cy-100:roi_cy+100, roi_cx-2*ray:roi_cx+2*ray]
    img_roicrop_rect = img_max_inverted[roi_cy-ray:roi_cy+ray, roi_cx-2*ray:roi_cx+2*ray]

    # corta uma ROI com centro no entroid em img_max_    
    if debug:
        cv2.imshow("6 - Definicao do Perimetro", img_roicrop_rect); 
        cv2.imwrite('roi/inicialization/7_marker_centroid_110ray.png', img_roicrop_rect)   
    
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
    img_max_[roi_cy-ray:roi_cy+ray, roi_cx-2*ray:roi_cx+2*ray] = img_max2_
    if debug:
        cv2.imshow("7 - Identificacao das duas componentes", img_max_);
        cv2.imwrite('roi/inicialization/8_two_largest_components.png', img_max_)
        
    #################################
    # 8 - Reconstrução morfológica das componentes
    #################################        
    img_max_ = img_max_ / 255
    img_max_inverted = img_max_inverted / 255

    
    #######################
    ## teste gradiente
    #######################
    #eed = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1], [1,1,1,1,1]])
    #dilated = cv2.dilate(img,eed)
    #eroded = cv2.erode(img,eed)
    #grad=dilated-eroded
    #if debug:
    #    cv2.imshow('Grad Dilate', dilated)
    #    cv2.imshow('Grad Erode', eroded)
    #    cv2.imshow('Gradiente', grad)
#
    #grad[grad<0.150] = 0
    #grad[grad>0.150] *= 1.10
    #if debug:
    #    cv2.imshow('Gradiente 2', grad)
    #plt.imshow(grad)
    #######################
    ## teste linha vertical
    #######################
    arr_idx_sum = {}
    mindetect = 0
    for y in np.arange(0, np.size(img_max_inverted,1), 10):
        _sum = np.sum(img_max_inverted[:,y])
        if _sum < 150:
            mindetect += 1
            #print(y,'- sum:', _sum)
            #img_max_inverted[:,y] = 127
            arr_idx_sum[y]=_sum
    if mindetect>1:
        sorted_x = sorted(arr_idx_sum.items(), key=operator.itemgetter(1))
        idx_pointB = int(mindetect/2)
        #print("idx_pointB: ", idx_pointB, ", mindetect: ", mindetect)
        #print('Teste!!!',sorted_x[idx_pointB][0], sorted_x[0][0])
        img_max_inverted[:, min(sorted_x[idx_pointB][0], sorted_x[0][0]) : max(sorted_x[idx_pointB][0], sorted_x[0][0])] = 0
        #print('sorted_x', sorted_x)
   
    #img_max_inverted[111:170,180:256] = 0
    #######################
    ## fim do teste linha vertical
    #######################
        
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

    #######################

    if debug:
        cv2.imshow("8 - Recontrucao Mask", img_max_inverted); 
        cv2.normalize(img_max_inverted, img_max_inverted, 0, 255, cv2.NORM_MINMAX)
        img_max_inverted = cv2.convertScaleAbs(img_max_inverted) 
        cv2.imwrite('roi/inicialization/9_reconstruction_mask.png', img_max_inverted)  
        cv2.imshow("8 - Recontrucao Result", img_max_);
        cv2.imwrite('roi/inicialization/9_reconstruction_result.png', img_max_)
    
    ee=np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    img_max_ = cv2.erode(img_max_, ee)
    ee=np.array([[1, 1, 1],[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    img_max_ = cv2.erode(img_max_, ee)
    
    if debug:
        cv2.imshow("Init", img_max_);
        cv2.imwrite('roi/inicialization/10_initialization.png', img_max_)
        cv2.waitKey(0)

    ee = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
            ])

    img_max_ = cv2.dilate(img_max_, ee)

    if debug:
        cv2.imshow("Init Dilated", img_max_);
        cv2.imwrite('roi/inicialization/11_initialization_dilate.png', img_max_)

    return img_max_,[]