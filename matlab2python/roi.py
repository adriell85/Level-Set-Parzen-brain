import numpy as np
import matplotlib.pyplot as plt
import cv2

countImg = 1
pathSamples = "../../../base de imagens/Ecocardiograma/Casos/Todos os Casos/"

img_samples = []
img_samples.append("CasoAD.bmp")
img_samples.append("CasoAS.bmp")
img_samples.append("CasoBD.bmp")
img_samples.append("CasoBS.bmp")
img_samples.append("CasoCD.bmp")
img_samples.append("CasoCS.bmp")
img_samples.append("CasoDD.bmp")
img_samples.append("CasoDS.bmp")
img_samples.append("CasoED.bmp")
img_samples.append("CasoES.bmp")
img_samples.append("CasoFD.bmp")
img_samples.append("CasoFS.bmp")
img_samples.append("CasoGD.bmp")
img_samples.append("CasoGS.bmp")
img_samples.append("CasoHD.bmp")
img_samples.append("CasoHS.bmp")
img_samples.append("CasoID.bmp")
img_samples.append("CasoIS.bmp")
img_samples.append("CasoJD.bmp")
img_samples.append("CasoJS.bmp")
img_samples.append("CasoKD.bmp")
img_samples.append("CasoKS.bmp")
img_samples.append("CasoLD.bmp")
img_samples.append("CasoLS.bmp")
img_samples.append("CasoMD.bmp")
img_samples.append("CasoMS.bmp")
img_samples.append("CasoND.bmp")
img_samples.append("CasoNS.bmp")
img_samples.append("CasoOD.bmp")
img_samples.append("CasoOS.bmp")
img_samples.append("CasoPD.bmp")
img_samples.append("CasoPS.bmp")
img_samples.append("CasoQD.bmp")
img_samples.append("CasoQS.bmp")

for i in np.arange(0, np.size(img_samples, 0)):
    img = cv2.imread(pathSamples + img_samples[i], 0)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    # ------------------------------
    # Binarizacao da imagem
    # ------------------------------
    ret, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY)

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
    # for i in np.arange(0, np.size(stats, 0)):
    #    print(str(i) +": "+str(stats[i, cv2.CC_STAT_AREA]))
    # print(stats.shape)

    # ------------------------------
    # Detecta e separa a maior componente
    # ------------------------------
    img_max = np.zeros_like(labels)
    idx_maxcomp = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()

    img_max[labels != idx_maxcomp] = 0
    img_max[labels == idx_maxcomp] = 255

    # ------------------------------
    # Desenha circulo - ROI
    # ------------------------------
    ray = 120
    roi_cx = int(centroids[idx_maxcomp, 0])
    roi_cy = int(centroids[idx_maxcomp, 1])
    img_roi = np.zeros_like(img)

    cv2.circle(img_roi, (roi_cx, roi_cy), ray, (1, 0, 0), -1)
    img_roicrop = img_roi * img

    # img_roicrop_rect = img[roi_cy-ray:roi_cy+ray, roi_cx-ray:roi_cx+ray]
    img_roicrop_rect = img_blur[roi_cy - ray:roi_cy + ray, roi_cx - ray:roi_cx + ray]

    # ------------------------------
    # detect circles in the image
    # img_roicrop_rect[img_roicrop_rect<25] = 127;
    # img_roicrop_rect[img_roicrop_rect<50] = 0;

    output = img_roicrop_rect.copy()
    gray = img_roicrop_rect.copy()
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 5, 70, 3, param1=10, param2=100, minRadius=50, maxRadius=100)

    print("Circles.shape:")
    print(type(circles))

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        print(circles.shape)

        idxDelete = []
        i = 0
        for (x, y, r) in circles:
            print("(x, y, r) = ({}, {}, {})".format(x, y, r))
            if x - r < 0 or x + r > 2 * ray:
                idxDelete.append(i)
            if y - r < 0 or y + r > 2 * ray:
                idxDelete.append(i)
                # if (x+r > roi_cx+ray) or (x-r < roi_cx-ray):
            #    idxDelete.append(i)
            # if (y+r > roi_cy+ray) or (y-r < roi_cy-ray):
            #    idxDelete.append(i)
            i = i + 1

        print("\ncircles.shape antes:")
        print(circles.shape)

        circles = np.delete(circles, idxDelete, 0)
        print("\ncircles.shape:")
        print(circles.shape)
        idxCircleLargest = circles[:, 2].argmax()
        print("\nidxCircleLargest:")
        print(idxCircleLargest)

        # loop over the (x, y) coordinates and radius of the circles
        i = 0
        for (x, y, r) in circles:
            print(str(i) + ": raio =" + str(r));

            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            if idxCircleLargest == i:
                print("Plot circle index" + str(i))
                cv2.circle(output, (x, y), r, (255, 0, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 0), -1)

            i = i + 1

    else:
        print("nenhum circulo detectado!")

    # show the output image
    cv2.imshow("output", np.hstack([img_roicrop_rect, output]))
    cv2.waitKey(0)

    # ------------------------------
    # Imprime imagens
    # ------------------------------
    # numPrintImgs = 4
    # plt.subplot(1,numPrintImgs,countImg), plt.imshow(img,'gray'); countImg+=1
    # plt.subplot(1,numPrintImgs,countImg), plt.imshow(img_bin,'gray'); countImg+=1
    # plt.subplot(1,numPrintImgs,countImg), plt.imshow(img_roicrop,'gray'); countImg+=1
    # plt.subplot(1,numPrintImgs,countImg), plt.imshow(255-img_roicrop_rect,'gray'); countImg+=1

    # plt.show()