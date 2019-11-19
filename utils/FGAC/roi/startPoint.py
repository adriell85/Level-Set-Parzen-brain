import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_gradient_magnitude

def getInitialPoint(img_path):
    iniX=200
    iniY=200
    
    img = cv2.imread(img_path,0) 

    #cv2.imshow("img",img)
    #cv2.waitKey(0)
    #img[img <= (50)] = 0;
    #img[img >= (200)] = 255;
    #img = cv2.cvtColor(img), cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img,(5,5),0)

    # ------------------------------
    # Binarizacao da imagem
    # ------------------------------
    ret, img_bin = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY)

    # ------------------------------
    # Identifica a maior componente
    # ------------------------------
    connectivity = 4  
    # Perform the operation
    output = cv2.connectedComponentsWithStats(img_bin, connectivity, cv2.CV_8U)

    # Rotulos das componentes identificadas
    num_labels = output[0]
    # Matriz com os pixels identificados com os rotulos
    labels = output[1]

    # Matriz Estatisticas calculadas para cada componente conexa detectada
    # stats[label, COLUMN]
    # cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
    # cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
    # cv2.CC_STAT_WIDTH The horizontal size of the bounding box
    # cv2.CC_STAT_HEIGHT The vertical size of the bounding box
    # cv2.CC_STAT_AREA The total area (in pixels) of the connected component
    stats = output[2]
    # Matrix com os centroides das componentes
    centroids = output[3]

    # Exibe area das componentes conexas identificadas
    #for i in np.arange(0, np.size(stats, 0)):
    #    print(str(i) +": "+str(stats[i, cv2.CC_STAT_AREA]))
    #print(stats.shape)

    # ------------------------------
    # Detecta e separa a maior componente
    # ------------------------------
    img_max = np.zeros_like(labels) 
    idx_maxcomp = 1+stats[1:, cv2.CC_STAT_AREA].argmax()

    img_max[labels != idx_maxcomp] = 0
    img_max[labels == idx_maxcomp] = 255

    # ------------------------------
    # Desenha circulo - ROI
    # ------------------------------
    ray = 120
    roi_cx = int(centroids[idx_maxcomp,0])
    roi_cy = int(centroids[idx_maxcomp,1])
    img_roi = np.zeros_like(img)

    cv2.circle(img_roi, (roi_cx, roi_cy), ray, (1,0,0), -1)
    img_roicrop = img_roi*img

    #img_roicrop_rect = img[roi_cy-ray:roi_cy+ray, roi_cx-ray:roi_cx+ray]
    img_roicrop_rect = img_blur[roi_cy-ray:roi_cy+ray, roi_cx-ray:roi_cx+ray]

    # ------------------------------
    # detect circles in the image
    #img_roicrop_rect[img_roicrop_rect<25] = 127;
    #img_roicrop_rect[img_roicrop_rect<50] = 0;
    stackImgShow = img_roicrop_rect
    output = img_roicrop_rect.copy()
    gray = img_roicrop_rect.copy()
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 5, 70, 3,param1=10,param2=100,minRadius=50,maxRadius=100)
    
    print("Circles.shape:")
    print(type(circles))

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int") 
        print(circles.shape)

        idxDelete=[]
        i=0
        for (x, y, r) in circles:
            print("(x, y, r) = ({}, {}, {})".format(x, y, r))
            if x-r < 0 or x+r > 2*ray:
                idxDelete.append(i)   
            if y-r < 0 or y+r > 2*ray:
                idxDelete.append(i)                
            #if (x+r > roi_cx+ray) or (x-r < roi_cx-ray):
            #    idxDelete.append(i)            
            #if (y+r > roi_cy+ray) or (y-r < roi_cy-ray):
            #    idxDelete.append(i) 
            i=i+1 

        print("\ncircles.shape => before:")
        print(circles.shape)
        circles = np.delete(circles, idxDelete, 0)
        print("\ncircles.shape => np.delete:")
        print(circles.shape)

        if circles is not None and np.size(circles, 0) > 0:
            
            idxCircleLargest = circles[:,2].argmax()
            print("\nidxCircleLargest:")
            print(idxCircleLargest)        

            # loop over the (x, y) coordinates and radius of the circles
            i=0
            sumPixelsNew=0
            sumPixelsOld=999999
            idxMoreDark=0            
            for (x, y, r) in circles:
                print(str(i)+": raio ="+str(r));
            
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                #if idxCircleLargest == i:
                print("Plot circle index" + str(i))
                mask = np.zeros((np.size(img_roicrop_rect,0),np.size(img_roicrop_rect,1)),dtype=np.uint8)
                #AM: Original:
                #cv2.circle(output, (x, y), r, (255, 0, 0), 4)
                cv2.circle(mask, (x, y), r, (1, 1, 1), -1,8,0)   
                miniCircularCrop = (mask)*img_roicrop_rect
                #sumPixelsNew=miniCircularCrop.sum()

                #AM: Identifica o círculo com região mais escura
                if sumPixelsNew<sumPixelsOld:
                   sumPixelsOld=sumPixelsNew
                   idxMoreDark=i

                stackImgShow = np.hstack([stackImgShow, miniCircularCrop])
                #cv2.imshow("roi-mini-circle",miniCircularCrop)
                #cv2.waitKey(0)
                
                #img_roicrop_rect
                #break;

                i = i+1

            #AM: A partir do circulo detectado, coleta x, y, c para plotar
            (x, y, r) = circles[idxMoreDark,:]
            cv2.circle(output, (x, y), r, (255, 0, 0), 4)
            
            #-------------------------------------
            #AM: Mini circulos de análise
            #-------------------------------------
            r_mini = int(r/3)
            circle_mini = []
            circle_mini.append([x-r_mini, y])             
            circle_mini.append([x+r_mini, y])             
            circle_mini.append([x, y-r_mini])             
            circle_mini.append([x, y+r_mini])             
            circle_mini.append([x-r_mini, y-r_mini])    
            circle_mini.append([x-r_mini, y+r_mini])    
            circle_mini.append([x+r_mini, y-r_mini])    
            circle_mini.append([x+r_mini, y+r_mini])    
            circle_mini.append([x, y])    

            #-----------------------------
            # Identifica o circulo inscrito no circulo maior cuja sobreposição
            # seja mais escura
            #-----------------------------
            iMini=0
            sumPixelsNewMini=0
            sumPixelsOldMini=999999
            idxMoreDarkMini=0            
            for (x,y) in circle_mini:
                mask = np.zeros((np.size(img_roicrop_rect,0),np.size(img_roicrop_rect,1)),dtype=np.uint8)                
                #cv2.circle(output, (x, y), r, (255, 0, 0), 4)
                cv2.circle(mask, (x, y), r_mini, (1, 1, 1), -1,8,0)   
                miniCircularCrop = (mask)*img_roicrop_rect
                sumPixelsNewMini=miniCircularCrop.sum()

                #AM: Identifica o círculo com região mais escura
                if sumPixelsNewMini<sumPixelsOldMini:
                   sumPixelsOldMini=sumPixelsNewMini
                   idxMoreDarkMini=iMini

                iMini=iMini+1

            print("idxMoreDarkMini: ")
            print(idxMoreDarkMini)
            print(np.size(circle_mini,0))
            #AM: A partir do mini circulo detectado, coleta x, y, c para plotar
            [x, y] = circle_mini[idxMoreDarkMini]
            cv2.circle(output, (x, y), r_mini, (127, 0, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (127, 0, 0), -1)

            print("Centro Big ROI:")    
            print([roi_cx, roi_cy])    

            print("Centro Little ROI:")    
            print([x, y])

            iniX=(roi_cx-ray) + x #+ (2*ray-x)
            iniY=(roi_cy-ray) + y #(2*ray-y)             
        else:
            print("nenhum circulo detectado!")  
            iniX=roi_cx
            iniY=roi_cy      
    else:
        print("nenhum circulo detectado!")
        iniX=roi_cx
        iniY=roi_cy
        # show the output image

    cv2.imshow("output", np.hstack([stackImgShow, output]))
    cv2.waitKey(0)
    #plt.title('Roi')
    #plt.imshow(output)  

    return [iniX, iniY]     

###########################################################################################

