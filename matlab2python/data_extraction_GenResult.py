# -*- coding: utf-8 -*-

from __future__ import division
import time
import cv2
import numpy as np
import natsort
import os
from cv2 import imwrite

def perform_measure(gac_image, doctor_result):
    from math import sqrt
    # from matplotlib import pyplot as ppl

    tp = 0
    fp = 0
    tn = 0
    fn = 0


    imSum = gac_image + doctor_result ;
    imDif = gac_image - doctor_result ;
    tp  =  np.float(np.sum((imSum == 2) * 1, dtype=np.int32))
    tn  = np.float(np.sum((imSum == 0) * 1, dtype=np.int32))
    fp  = np.float(np.sum((imDif < 0) * 1, dtype=np.int32))
    fn = np.float(np.sum((imDif > 0) * 1, dtype=np.int32))


    try:
        precision = tp / (tp + fp)
    except:
        precision = 0

    try:
        recall = tp / (tp + fn)
    except:
        recall = 0

    try:
        dice = (2 * tp) / ((2 * tp) + fp + fn)
    except:
        dice = 0

    try:
        sens = tp / (tp + fn)
    except:
        sens = 0
    try:
        spec = tn / (tn + fp)
    except:
        spec = 0
    try:
        accu = (tp + tn) / (tp + tn + fp + fn)
    except:
        accu = 0
    try:
        f1sc = 2 * ((precision * recall) / (precision + recall))
    except:
        f1sc = 0
    try:
        matt = ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    except:
        matt = 0


    jacc = jaccard_coefficient(gac_image, doctor_result)

    return [dice, sens, spec, accu, f1sc, matt,jacc]


def jaccard_coefficient(gac_image, doctor_result):
    from sklearn.metrics import jaccard_similarity_score as js

    return js(gac_image,doctor_result)


def conformity_coefficient(gac_image, doctor_result):
    conformity_result = np.corrcoef(np.ravel(gac_image), np.ravel(doctor_result))
    return conformity_result[0, 1]


def hausdorff_dist(gac_image, doctor_result):
    from scipy.spatial.distance import directed_hausdorff
    u = directed_hausdorff(doctor_result, gac_image)[0]
    v = directed_hausdorff(gac_image, doctor_result)[0]
    return max(u, v)


def return_measures(final_image, doctor_image):
    haus = hausdorff_dist(final_image, doctor_image)
    jacc = jaccard_coefficient(final_image, doctor_image)
    conf = conformity_coefficient(final_image, doctor_image)
    dice, sens, spec, accu, fscr, matt = perform_measure(final_image, doctor_image)
    return fscr, accu, dice, jacc, matt, sens, spec, conf, haus


def save_file_measures(final_image, doctor_image, image_initialization):
    haus = hausdorff_dist(final_image, doctor_image)
    jacc = jaccard_coefficient(final_image, doctor_image)
    conf = conformity_coefficient(final_image, doctor_image)
    dice, sens, spec, accu, fscr, matt = perform_measure(final_image, doctor_image)

    file = open("results/file_results/{}.txt".format(image_initialization), "w+")
    file.write('\tImage {}.dcm\n\n'.format(image_initialization))
    file.write('F-score                 = {}\n'.format(fscr))
    file.write('Accuracy score          = {}\n'.format(accu))
    file.write('Dice coefficient        = {}\n'.format(dice))
    file.write('Hausdorff distance      = {}\n'.format(haus))
    file.write('Jaccard coefficient     = {}\n'.format(jacc))
    file.write('Matthews coefficient    = {}\n'.format(matt))
    file.write('Conformity coefficient  = {}\n'.format(conf))
    file.write('Sensitivity coefficient = {}\n'.format(sens))
    file.write('Specificity coefficient = {}\n'.format(spec))
    file.close()


def print_measures(final_image, doctor_image):
    h = hausdorff_dist(final_image, doctor_image)
    j = jaccard_coefficient(final_image, doctor_image)
    c = conformity_coefficient(final_image, doctor_image)
    d, s, p, a, f, m = perform_measure(final_image, doctor_image)

    print('')
    print('F-score                 = %f' % f)
    print('Accuracy score          = %f' % a)
    print('Dice coefficient        = %f' % d)
    print('Hausdorff distance      = %f' % h)
    print('Jaccard coefficient     = %f' % j)
    print('Matthews coefficient    = %f' % m)
    print('Conformity coefficient  = %f' % c)
    print('Sensitivity coefficient = %f' % s)
    print('Specificity coefficient = %f' % p)


def get_multithresholding_image(gray_image, t_min, t_max):
    thresholding_image = np.zeros(gray_image.shape)
    for i in range(0, gray_image.shape[0]):
        for j in range(0, gray_image.shape[1]):
            thresholding_image[i][j] = 1 if ((gray_image[i][j] >= t_min) & (gray_image[i][j] < t_max)) else 0
    return thresholding_image


def normalizeImage(v):
    v = (v - v.min()) / (v.max() - v.min())
    result = (v * 255).astype(np.uint8)
    return result

path_folder_gt = "../datasets/resultados_GT"
path_folder_seg = "../datasets/resultsA"

metrics_imgs = []

for dirName, subdirList, filelist in os.walk(path_folder_gt):
    dirList = natsort.natsorted(filelist, reverse=False)


    for file in dirList:
        gt = cv2.imread(os.path.join(path_folder_gt, file), 0)

        gt=np.uint8(gt)
        img_gt = (gt > 100) * 1

        res = cv2.imread(os.path.join(path_folder_seg, file), 0)
        res = cv2.resize(res, (512, 512))
        print(np.shape(res), np.shape(gt))
        img_seg = (res > 100) * 1
        img4 = normalizeImage(img_gt - img_seg)
        vis = np.concatenate((gt, res), axis=1)
        # cv2.imshow('1y32', vis)
        # k = cv2.waitKey(0)

        metrics_imgs.append(perform_measure(img_seg, img_gt))


# print(np.shape(metrics_imgs))
# print(metrics_imgs[:][0])

dice = []
sens = []
spec = []
accu = []
f1sc = []
matt = []
jacc = []

for metrics in metrics_imgs:
    dice.append(metrics[0])
    sens.append(metrics[1])
    spec.append(metrics[2])
    accu.append(metrics[3])
    f1sc.append(metrics[4])
    matt.append(metrics[5])
    jacc.append(metrics[6])


print('dice: {0:.4f} +/- {1:.4f}\nsens: {2:.4f} +/- {3:.4f}\nspec: {4:.4f} +/- {5:.4f}\naccu: {6:.4f} +/- {7:.4f}\nf1sc: {8:.4f} +/- {9:.4f}\nmatt: {10:.4f} +/- {11:.4f}\njacc: {12:.4f} +/- {13:.4f}'.format(np.mean(dice),
                                                                                                                                                       np.std(dice),np.mean(sens),np.std(sens),np.mean(spec),np.std(spec),
                                                                                                                                                       np.mean(accu),np.std(accu),np.mean(f1sc),np.std(f1sc),np.mean(matt),
                                                                                                                                                       np.std(matt),np.mean(jacc),np.std(jacc)))
# salvar em arquivo
file = open('../datasets/resultsA/results.txt','w')
file.write('\ndice: {0:.4f} +/- {1:.4f}\nsens: {2:.4f} +/- {3:.4f}\nspec: {4:.4f} +/- {5:.4f}\naccu: {6:.4f} +/- {7:.4f}\nf1sc: {8:.4f} +/- {9:.4f}\nmatt: {10:.4f} +/- {11:.4f}\njacc: {12:.4f} +/- {13:.4f}'.format(np.mean(dice),
                                                                                                                                                       np.std(dice),np.mean(sens),np.std(sens),np.mean(spec),np.std(spec),
                                                                                                                                                       np.mean(accu),np.std(accu),np.mean(f1sc),np.std(f1sc),np.mean(matt),
                                                                                                                                                       np.std(matt),np.mean(jacc),np.std(jacc)))
file.close()


