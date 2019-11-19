import numpy as np
import math
from scipy import linalg
import cv2


# Parzen that generates a segmentation basen on Parzen Window
class ParzenWindow:

    # TODO - Increase the number of classes allowed.
    #        In this version only 2 classes is possible.

    # Define the constructor of the class
    def __init__(self, image, lesion=[], background=[], inf_thresh=0, sup_thresh=255, h=0.8, n_points=15):
        self.image = image
        self.inf_thresh = inf_thresh
        self.sup_limit = sup_thresh

        # If the points of lesion and background is not passed, then find the first n points
        if not(lesion or background):
            self.lesion, self.background, self.all_pixels, self.all_classes = \
                self.__find_roi(inf_thresh, sup_thresh, n_points)

        else:
            self.lesion = lesion
            self.background = background

        self.num_points = n_points
        self.h = h

    # Find n points to be the lesion and background points
    def __find_roi(self, inf_thresh, sup_thresh, n_points):

        concatenated_pixels = []
        all_classes = []

        for pixel in np.nditer(self.image):
            concatenated_pixels.append(pixel)

        pass_size = int((sup_thresh - inf_thresh) / n_points)

        lesion = list(range(inf_thresh, sup_thresh, pass_size))

        if inf_thresh <= 30:
            background = list(range(sup_thresh + 10, 230, pass_size))
        else:
            background = list(range(0, inf_thresh - 10, pass_size*2))
            background += list(range(sup_thresh + 10, 230, pass_size))

        all_classes.append(lesion)
        all_classes.append(background)

        return lesion, background, concatenated_pixels, all_classes

    def segmentation(self, n_classes=2):
        number_of_pixels = self.image.size
        p = []
        for i in range(n_classes):
            for j in range(number_of_pixels):
                p.append(((1 / (1 * math.pi * self.h * self.h) ** (1 / 2)) *
                         math.exp((-1 / (2 * self.h * self.h)) *
                         linalg.norm(self.all_pixels[j] - self.all_classes[i]))) * len(self.all_classes[i]))

        col_1 = p[:number_of_pixels]
        col_2 = p[number_of_pixels:]

        segment = []
        for i, j in zip(col_1, col_2):
            if i > j:
                segment.append(1)
            else:
                segment.append(2)

        rows, cols = self.image.shape
        img_new = self.image.copy()

        cont = 0
        for i in range(rows):
            for j in range(cols):
                img_new[i, j] = segment[cont]
                if img_new[i, j] == 1:
                    img_new[i, j] = 0
                else:
                    img_new[i, j] = 255
                cont += 1

        return img_new
