import numpy as np
import matplotlib.pyplot as plt
import cv2
from startPoint import getInitialPoint

countImg=1 
pathSamples="../../../base de imagens/Ecocardiograma/Casos/Todos os Casos/"

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


#img_samples = []
#img_samples.append("CasoBD.bmp")

for i in np.arange(0,np.size(img_samples,0)):
    img_path=pathSamples+img_samples[i]
    getInitialPoint(img_path)