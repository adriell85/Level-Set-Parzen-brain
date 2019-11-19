from data_information import dcm_information as di
from skimage.segmentation import clear_border
# from withinSkull import withinSkull
# from mls_parzen import mls_parzen, conv2
# import FolClustering as fl
# from AreaLabel import AreaLabel
from cv2 import connectedComponentsWithStats, CV_8U
from numba import njit
from numba import jit
import numpy as np
import cv2
from scipy.spatial import distance
import pywt

def within_skull(image):
    img_oss = np.where(image > 100, 0, 1)
    img_oss = np.uint8(img_oss)
    img_oss_ant = np.where(img_oss < 1, 1, 0)
    img_oss = np.uint8(img_oss)
    img_oss_ant = np.uint8(img_oss_ant)
    img_oss_ant = np.where(img_oss_ant < 1, 1, 0)
    img_oss_ant = np.uint8(img_oss_ant)
    return img_oss, img_oss_ant


def image_morfologic1(image, Kernel, iterations):
    if Kernel == 1:
        kernel = np.asarray([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]], np.uint8)
    if Kernel == 2:
        kernel = np.asarray([[0, 1, 1, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [0, 1, 1, 1, 1, 1, 0]], np.uint8)
    img_oss_erode = cv2.erode(image, kernel, iterations=iterations)

    img_oss = clear_border(img_oss_erode)
    return img_oss, img_oss_erode


@njit
def mult(img_res, img_oss, img_oss_erode):
    img_si = np.multiply(img_res, img_oss)
    # img_si = img_res * img_oss

    # img_si = img_si * img_oss_erode
    img_si=np.multiply(img_si,img_oss_erode)
    return img_si, img_si, img_si


def thresolded_img(img_si_ant):
    img_si_ant = np.uint8(img_si_ant)
    img_si_ant = cv2.equalizeHist(img_si_ant)
    # img_orig_log = img_si_ant
    ret, img_si_ant = cv2.threshold(img_si_ant, 227, 255, type=cv2.THRESH_BINARY)
    img_si_ant_median = img_si_ant
    img_si_ant = cv2.medianBlur(img_si_ant, 3)
    return (img_si_ant, img_si_ant_median)


def image_morfologic2(img_si_ant):
    kernel1 = np.asarray([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], np.uint8)

    kernel2 = np.asarray([[0, 1, 1, 1, 1, 1, 0],
                          [1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 1, 1, 0]], np.uint8)
    img_si_after_erode = img_si_ant
    # for it in range(8):
    #     img_si_ant = cv2.erode(img_si_ant, kernel1)
    #     img_si_ant = cv2.dilate(img_si_ant, kernel1)
    img_si_ant=cv2.morphologyEx(img_si_ant,cv2.MORPH_OPEN,kernel1)
    # for it in range(8):
    #     img_si_ant = cv2.erode(img_si_ant, kernel2)
    #     img_si_ant = cv2.dilate(img_si_ant, kernel2)
    img_si_ant = cv2.morphologyEx(img_si_ant, cv2.MORPH_OPEN, kernel2)
    return img_si_ant, img_si_after_erode


def fors1(n, centroids, img_oss_center):
    # for i in range(n):
    #     # a = (centroids[i, 1], centroids[i, 0])
    #
    #     distances[i] = distance.euclidean(img_oss_center, (centroids[i, 1], centroids[i, 0]))

    distances=[(distance.euclidean(img_oss_center, (centroids[i, 1], centroids[i, 0]))) for i in range(n) ]
    # val1 = [i for i in range(n) if distances[i] == distance1]
    return distances



def fors2(n, areas, distances):

    # for k in range(n):
    #     if areas[k] < 60:
    #         distances[k] = 1000000
    #
    # dists=[10000 for k in range(n) if areas[k]<60]
    # ================================== parei aqui====================================================================
    areas=np.uint32(areas)
    ars=np.where(areas<60,10000,1)
    # img_oss=np.where(img_res>100,0,1)
    ars=np.int32(ars)
    distances=np.multiply(distances,ars)





    # distances = [1000000 for k in range(n) if areas[k] < 60]

    # [x + 1 for x in l if x >= 45 else x + 5]
    # area = [sum(l == x) for x in range(0, output[0])]  # Number of labels



    distances_org = np.sort(distances)

    distance1 = distances_org[1]
    val1=[i for i in range(n) if distances[i]==distance1]
    # for i in range(1, n):
    #     if distances[i] == distance1:
    #         val1 = i
    return val1


def centroid_operations1(centroids, img_oss_center, stats, n, labels):
    # distances = [0] * n
    # distances=np.zeros(n)

    # for i in range(n):
    #     a = (centroids[i, 1], centroids[i, 0])
    #
    #     distances[i] = distance.euclidean(img_oss_center, a)
    distances = fors1(n, centroids, img_oss_center)

    # # Loop through areas in order of size
    areas = [s[4] for s in stats]
    # sorted_idx = np.argsort(np.unique(areas))
    # areas_org = np.sort(areas)

    val1 = fors2(n, areas, distances)
    # for k in range(n):
    #     if areas[k] < 60:
    #         distances[k] = 1000000
    # distances_org = np.sort(distances)
    #
    # distance1 = distances_org[1]
    #
    # for i in range(1, n):
    #     if distances[i] == distance1:
    #         val1 = i

    img_filtered = np.where((labels == val1), 255, 0)
    img_filtered = np.uint8(img_filtered)
    return img_filtered


def fors3(n, centroids,img_filtered_center):
    # for i in range(n):
    #     a = (centroids[i, 1], centroids[i, 0])
    #
    #     distances[i] = distance.euclidean(img_filtered_center, a)
    distances = [(distance.euclidean(img_filtered_center, (centroids[i, 1], centroids[i, 0]))) for i in range(n)]
    distances=np.argsort(distances)
    val1=distances[0]
    # smallest_distance = distances[0]

    # for i in range(1, n):
    #     if smallest_distance > distances[i]:
    #         val1 = i
    #         smallest_distance = distances[i]
    # smallest_distance=[distances[i] for i in range(n) if distances[i]<smallest_distance]
    # val1 = [i for i in range(len(smallest_distance)) if smallest_distance[i] >= sml]
    return val1


def centroid_operations2(centroids, img_filtered_center, labels, n):
    # distances = [0] * n

    # for i in range(n):
    #     a = (centroids[i, 1], centroids[i, 0])
    #
    #     distances[i] = distance.euclidean(img_filtered_center, a)
    # smallest_distance = distances[0]
    #
    # for i in range(1, n):
    #     if smallest_distance > distances[i]:
    #         val1 = i
    #         smallest_distance = distances[i]
    val1 = fors3(n, centroids,img_filtered_center)

    img_final = np.where((labels == val1), 255, 0)
    img_final = np.uint8(img_final)
    return img_final

# def haar(img):
#  # coeficients=pywt.idwt2(img,'haar')
#  img=np.float64(img)
#
#  coeffs = pywt.dwt2(img, 'haar')
#  # pywt.dwt2(coeffs, 'haar')
#  cA, (cH, cV, cD) = coeffs
#
#  return cA,cH,cV,cD
#
# def anti_haar(cA,cH,cV,cD):
#  # coeficients=pywt.idwt2(img,'haar')
#  cA=np.float64(cA)
#  cH = np.float64(cH)
#  cV = np.float64(cV)
#  cD = np.float64(cD)
#
#  # pywt.idwt2(cA,cH,cV,cD, 'haar')
#  # coeffs = pywt.dwt2(data, 'haar')
#  # pywt.idwt2(coeffs, 'haar')
#  # pywt.dwt2(coeffs, 'haar')
#  # coeffs=cA, (cH, cV, cD)
#
#  return pywt.idwt2((cA,(cH,cV,cD)), 'haar')
