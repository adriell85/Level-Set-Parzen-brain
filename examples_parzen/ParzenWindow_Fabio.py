# Parzen that generates a segmentation basen on Parzen Window
class ParzenWindow:

    # Define the constructor of the class
    def __init__(self, file, image):
        self.image = image
        self.h, self.n_points, self.n_classes, self.classes = self.get_parzen_config(file)

        self.all_pixels, self.all_classes = self.__find_roi()

    # Find n points to be the lesion and background points
    def __find_roi(self):

        concatenated_pixels = self.image.ravel()

        all_classes = []
        for key in self.classes:
            try:
                inf_limit, sup_limit = self.classes[key].split('-')
            except ValueError:
                print('Interval missing in one or more class.')
                exit(0)

            pass_size = int((int(sup_limit) - int(inf_limit)) / self.n_points)

            try:
                interval = range(int(inf_limit), int(sup_limit), pass_size)
            except ValueError:
                interval = range(int(inf_limit), int(sup_limit))

            self.classes[key] = [x for x in interval]

            all_classes.append(self.classes[key])

        return concatenated_pixels, all_classes

    def segmentation(self):
        import math
        from scipy import linalg
        import numpy as np
        import pandas as pd

        number_of_pixels = self.image.size
        probability = 0

        for i in range(self.n_classes):
            p = []
            for j in range(number_of_pixels):
                p.append(((1 / (1 * math.pi * self.h * self.h) ** (1 / 2)) *
                          math.exp((-1 / (2 * self.h * self.h)) *
                                   linalg.norm(self.all_pixels[j] - self.all_classes[i]))) * len(self.all_classes[i]))

            if i == 0:
                probability = np.hstack([p])
            else:
                probability = np.vstack([probability, p])

        # Create a pandas DataFrame from the probabilities calculated
        df = pd.DataFrame(probability)

        # Find the lines (classes) that contain the greater probability in each column
        segment = df.idxmax()

        # Count the number of pixels in each class an put them in a list
        classes = []
        for i in range(1, self.n_classes):
            classes.append(np.count_nonzero(segment == i))

        # Count the total of pixels in the image, except the background
        total = np.count_nonzero(segment != 0)

        # Create a list of the percentages of each class in the image (ABTD)
        probability = [count_classes / total for count_classes in classes]

        return probability

        # TODO - make a function to return the segmentation of the image
        # segment = []
        # for i in range(number_of_pixels):
        #     segment.append(df[str(i)].argmax())

        # col_1 = p[:number_of_pixels]
        # col_2 = p[number_of_pixels:]

        # segment = []
        # for i, j in zip(col_1, col_2):
        #     if i > j:
        #         segment.append(1)
        #     else:
        #         segment.append(2)

        # rows, cols = self.image.shape
        # img_new = self.image.copy()
        #
        # cont = 0
        # for i in range(rows):
        #     for j in range(cols):
        #         img_new[i, j] = segment[cont]
        #         if img_new[i, j] == 0:
        #             img_new[i, j] = 0
        #
        #         elif img_new[i, j] == 1:
        #             img_new[i, j] = 30
        #
        #         elif img_new[i, j] == 2:
        #             img_new[i, j] = 60
        #
        #         elif img_new[i, j] == 3:
        #             img_new[i, j] = 90
        #
        #         elif img_new[i, j] == 4:
        #             img_new[i, j] = 120
        #
        #         elif img_new[i, j] == 5:
        #             img_new[i, j] = 150
        #
        #         elif img_new[i, j] == 6:
        #             img_new[i, j] = 180
        #
        #         elif img_new[i, j] == 7:
        #             img_new[i, j] = 210
        #
        #         cont += 1
        #
        # return img_new

    def get_parzen_config(self, filename):
        from configparser import SafeConfigParser
        from collections import OrderedDict

        with open(filename) as infile:
            config = infile.read()

            parser = SafeConfigParser()
            parser.read('parzen.ini')

            h = float(parser.get('parameters', 'h'))
            n_points = int(parser.get('parameters', 'n_points'))
            n_classes = int(parser.get('parameters', 'n_classes'))

            classes = {}

            for i in range(n_classes):
                classes[str(i)] = parser.get('classes', str(i))

            classes = OrderedDict(sorted(classes.items(), key=lambda t: t[0]))

            return h, n_points, n_classes, classes
