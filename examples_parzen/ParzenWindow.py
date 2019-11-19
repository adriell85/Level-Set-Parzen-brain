DEBUG = False


class ParzenWindow:
    """
    Parzen that generates a segmentation basen on Parzen Window
    """

    # TODO - Increase the number of classes allowed.
    #        In this version only 2 classes is possible.

    # Define the constructor of the class
    def __init__(self, image, lesion, background,
                 inf_thresh=0, sup_thresh=67, h=0.7, n_points=15):
        self.image = image
        self.inf_thresh = inf_thresh
        self.sup_limit = sup_thresh
        self.num_points = n_points
        self.h = h

        # If the points of lesion and background is not passed,
        # then find the first n points
        if not (lesion or background):
            self.lesion, self.background, self.all_pixels, self.all_classes = \
                self.__find_roi(inf_thresh, sup_thresh, n_points)

        else:
            self.lesion = lesion
            self.background = background

    def __find_roi(self, inf_thresh, sup_thresh, n_points):
        """
        Find n points to be the lesion and background points
        :param inf_thresh: int
        :param sup_thresh: int
        :param n_points: int
        :return: list
        """
        from numpy import nditer
        import numpy as np

        lesion = []
        background = []
        all_classes = []

        # iterate over image array
        # if pixel is in this threshold
        # if pixel is bigger of the array of lesion
        # append lesion array with pixel
        # if pixel if different of zero and background
        # is smaller than n points
        # append background array with pixel
        for pixel in nditer(self.image):
            if inf_thresh <= pixel <= sup_thresh and len(lesion) < n_points:
                lesion.append(pixel)
            elif pixel != 0 and len(background) < n_points:
                background.append(pixel)
            elif len(background) == n_points and len(lesion) == n_points:
                break

        if DEBUG:
            print('lesion', lesion)
            print('background', background)

        all_classes.append(lesion)
        all_classes.append(background)

        # append all pixels in a new array
        img_array = np.asarray(self.image).reshape(-1)

        return lesion, background, img_array, all_classes

    def segmentation(self, n_classes=2):
        """
        Perform image segmentation
        :param self:
        :param n_classes: int
        :return: np.array
        """
        number_of_pixels = self.image.size

        class_1, class_2 = self._pdf_multivariate_gauss(number_of_pixels, n_classes)

        if DEBUG:
            print('all pixels shape =', len(self.all_pixels))
            print('all classes shape =', len(self.all_classes))
            print('img size', number_of_pixels)
            print('col 1 shape =', len(class_1))
            print('col 2 shape =', len(class_2))

        return self._segment_classes(class_1, class_2)

    def _pdf_multivariate_gauss(self, number_of_pixels, n_classes):
        """
        Calculate the multivariate normal density (pdf)
        :param number_of_pixels: int
        :param n_classes: int
        :return: np.array, np.array
        """
        from math import sqrt
        from scipy.linalg import norm
        from numpy import pi, exp, zeros

        classes_shape = len(self.all_classes[0])

        part1 = 1 / sqrt(2 * pi) * pow(self.h, 2)
        part2 = -1 / (2 * self.h * self.h)

        p_x = zeros(number_of_pixels * n_classes, dtype=int)
        counter = 0
        for i in range(n_classes):
            for j in range(number_of_pixels):
                part3 = part2 * norm(self.all_pixels[j] - self.all_classes[i])
                p_x[counter] = (part1 * exp(part3) * classes_shape)
                counter += 1

        # separate column 1 and 2 by respective classes, as are currently 2,
        # separate by first class and second class
        return p_x[:number_of_pixels], p_x[number_of_pixels:]

    def _segment_classes(self, class_1, class_2):
        """
        Segment by classes
        :param class_1: np.array
        :param class_2: np.array
        :return: np.ndarray
        """
        from numpy import reshape, uint8

        # Separate images in depending of the biggest values of each class
        segment = [0 if i > j else 255 for i, j in zip(class_1, class_2)]
        # Reshape image to create a matrix (or a image)
        # Pass type as int of 8 bits
        return reshape(segment, newshape=self.image.shape).astype(uint8)


def maior_comp(image):
    from cv2 import connectedComponentsWithStats, CV_8U, CC_STAT_AREA
    from numpy import zeros

    connectivity = 4
    output = connectedComponentsWithStats(image, connectivity, CV_8U)
    labels = output[1]
    stats = output[2]
    img_max_ = zeros(image.shape, image.dtype)
    largecomponent1 = 1 + stats[1:, CC_STAT_AREA].argmax()
    img_max_[labels == largecomponent1] = 255
    return img_max_
