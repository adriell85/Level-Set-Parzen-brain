import numpy as np
import matplotlib.pyplot as plt
import cv2

def SetInitialPoint(image):
    blur = image

    cv2.imshow('Initial Image', blur)
    cv2.waitKey(1)


    for i in range(9): 
        blur = cv2.GaussianBlur(blur, (7,7), 0)

    cv2.imshow('Blur Image', blur)
    cv2.waitKey(1)

    ret, processedImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow('Binary Image', processedImage)
    cv2.waitKey(1)
    
    kernel = np.ones((11,11),np.uint8)
    erosion = cv2.erode(processedImage,kernel,iterations = 3)

    cv2.imshow('Processed Image', erosion)
    cv2.waitKey(0)

    #================================== MASK ==================================

    mask_img = cv2.imread('D:/Metodo Geodesico/GAC-Fontes/testimages/mask.png', 0)

    mask_img = cv2.resize(mask_img, (0,0), fx=0.4, fy=0.4)
    mask_img = cv2.resize(mask_img, (processedImage.shape[1], processedImage.shape[0]))

    result = mask_img & erosion

    cv2.imshow('Result', result)
    cv2.waitKey(0)

    #=============================================================================

    ######### CONNECTED COMPONENTS #############
    connectivity = 4
    output = cv2.connectedComponentsWithStats(result, connectivity, cv2.CV_8U)
    ############################################

    labels = output[1]   
    stats = output[2]
    centroids = output[3]
    '''
    print(labels)
    print(centroids)
    
    print(centroids[1, 0])

    if (centroids[1, 0] > 100):
        print('True')

    roi = np.zeros(processedImage.shape, processedImage.dtype)
    contour_1 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    #print(contour_1)
    #print(centroids[contour_1])

    #stats[contour_1, cv2.CC_STAT_AREA] = contour_1
    #contour_2 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    #print(contour_2)
    #print(centroids[contour_2])


    roi[labels == contour_1] = 255
    roi[labels != contour_1] = 0    
    
    '''
    for label in range(1,labels.shape[0]):
        contour = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
        #print(centroids[contour, 0])
        #print(centroids[contour, 1])

        if centroids[contour, 0] > 25  and centroids[contour, 0] < 260 and centroids[contour, 1] > 25 and centroids[contour, 1] < 260:
            roi = np.zeros(processedImage.shape, processedImage.dtype)
            
            roi[labels == contour] = 255
            roi[labels != contour] = 0 

            cv2.imshow("Inicialization", roi)
            cv2.waitKey(0)
            return roi

        else:
            stats[contour, cv2.CC_STAT_AREA] = contour
    
    #Comparar com os cantos para ignorar contornos achados nas bordas.
    '''
    cv2.imshow("ROI", roi)
    cv2.waitKey(1)
    
    return roi

    ######### FIND CONTOURS #########
    
    contours = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1]
    
    #################
    
    thresh = 150
    flag = True

    startPointX = 0
    startPointY = 0

    while flag:   
        for c in contours:
            M = cv2.moments(c)
            if  M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                #if cX > 40 and cX < 500 and cY > 40 and cY < 500:
                if cX > 20 and cX < 285 and cY > 20 and cY < 285:
                    cv2.drawContours(image, [c], -1, (255, 0, 0), 2)
                    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
                    cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    startPointX = cX
                    startPointY = cY
                    flag = False

        if startPointX == 0 and startPointY == 0:
                ret, processedImage = cv2.threshold(blur,thresh,255,cv2.THRESH_BINARY_INV)    
                contours = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[1]
                kernel = np.ones((5,5),np.uint8)
                erosion = cv2.erode(processedImage,kernel,iterations = 3)
                thresh += 10    

    cv2.imshow('Points detected', image)
    cv2.waitKey(1)
    
    return [startPointX, startPointY]
    '''

def choose_channel(src, channel):
    bgr = []
    blue, green, red = cv2.split(src)
    bgr.append(blue)
    bgr.append(green)
    bgr.append(red)

    return bgr[channel]

def mean_pixels(src):
    return np.sum(src)/(src.shape[0]*src.shape[1])

def normalize_image(src, meanPixels):
    mean_2 = mean_pixels(src)
    np.array(src) * (meanPixels/mean_2)
    
    return src

def return_original_size(src, rows, cols):
    return cv2.resize(src, (cols, rows))