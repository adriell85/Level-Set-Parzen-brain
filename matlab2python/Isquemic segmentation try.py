import  numpy, shutil, os, nibabel
import sys, getopt
import cv2
import natsort
from data_information import dcm_information as di
from matplotlib import pyplot as ppl
import scipy.misc
import numpy as np
def normalizeImage(v):
    v = (v - v.min()) / (v.max() - v.min())
    result = (v * 255).astype(np.uint8)
    return result

# cont=0
# cont1=0
# cont2=0




direct = 'C:/Users/gomes/Desktop/ISLES2017_Training'
path_folder = os.listdir(direct)
path_folder = natsort.natsorted(path_folder, reverse=False)

for folder in path_folder:
        path = os.path.join(direct, folder)
        path_enter=os.listdir(path)
        path_enter=natsort.natsorted(path_enter,reverse=False)
        for arq in path_enter:
              val=os.path.join(path,arq)
              val_enter=os.listdir(val)
              cont = 0
              cont1 = 0
              cont2 = 0
              for filename in val_enter:
                  if 'png' in filename.lower():
                    cont += 1
                    output = list(map(int, str(cont)))
                    if cont <10:
                        cont1 = 0
                        cont2 += 1
                    else:

                        cont1=output[0]
                        cont2=output[1]




                    input1 = direct+'/'+folder+'/VSD.Brain.XX.O.MR_ADC.128020/VSD.Brain.XX.O.MR_ADC.128020_z0'+str(cont1)+str(cont2)+'.png'
                    input2 = direct + '/' + folder + '/VSD.Brain.XX.O.MR_MTT.127014/VSD.Brain.XX.O.MR_MTT.127014_z0' + str(cont1)+str(cont2)+'.png'
                    input3 = direct + '/' + folder + '/VSD.Brain.XX.O.MR_rCBF.127016/VSD.Brain.XX.O.MR_rCBF.127016_z0' + str(cont1)+str(cont2)+'.png'
                    input4 = direct + '/' + folder + '/VSD.Brain.XX.O.MR_rCBV.127017/VSD.Brain.XX.O.MR_rCBV.127017_z0' + str(cont1)+str(cont2)+'.png'
                    input5 = direct + '/' + folder + '/VSD.Brain.XX.O.MR_Tmax.127018/VSD.Brain.XX.O.MR_Tmax.127018_z0' + str(cont1)+str(cont2)+'.png'
                    input6 = direct + '/' + folder + '/VSD.Brain.XX.O.MR_TTP.127019/VSD.Brain.XX.O.MR_TTP.127019_z0' + str(cont1)+str(cont2)+'.png'
                    input7 = direct + '/' + folder + '/VSD.Brain.XX.O.OT.128050/VSD.Brain.XX.O.OT.128050_z0' + str(cont1)+str(cont2)+'.png'

                    # output= 'C:/Users/gomes/Desktop/training translate/'+folder+'/'+arq
                    img1 = cv2.imread(input1,0)
                    img2 = cv2.imread(input2, 0)
                    img3 = cv2.imread(input3, 0)
                    img4 = cv2.imread(input4, 0)
                    img5 = cv2.imread(input5, 0)
                    img6 = cv2.imread(input6, 0)
                    img7 = cv2.imread(input7, 0)

                    cv2.imshow('img1',img1)
                    cv2.imshow('img2', img2)
                    cv2.imshow('img3', img3)
                    cv2.imshow('img4', img4)
                    cv2.imshow('img5', img5)
                    cv2.imshow('img6', img6)
                    cv2.imshow('img7', img7)
                    cv2.waitKey(100)
                    print(cont)


