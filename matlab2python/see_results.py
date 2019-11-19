from data_information import dcm_information as di
from matplotlib import pyplot as ppl
import cv2
image=cv2.imread('C:/Users/gomes/Desktop/VSD.Brain.XX.O.MR_4DPWI.129320.dicom/VSD.Brain.XX.O.MR_4DPWI.129320_t001_z001.png')
count_debug = di.show_figures(image, 1)
ppl.show()