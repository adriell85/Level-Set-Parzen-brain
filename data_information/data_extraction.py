class PerformMeasure(object):
    """
    Perform measures between segmented image and ground truth
    """

    def __init__(self, gac_img, doctor_result, time):
        self.precision = 0
        self.recall = 0
        self.dice = 0
        self.sens = 0
        self.spec = 0
        self.jacc = 0
        self.accu = 0
        self.f1sc = 0
        self.matt = 0
        self.conf = 0
        self.haus = 0
        self.set_convolution_matrix(gac_img, doctor_result)
        self.time = time
        self.df = self.convert_to_pandas(time)

    def set_time(self, time):
        self.time = time

    def set_convolution_matrix(self, gac_img, doctor_result):
        from math import sqrt
        from numpy import ravel, corrcoef
        from scipy.spatial.distance import directed_hausdorff
        import numpy as np
        # Sum and subtract two images
        img_sum = np.add(gac_img, doctor_result)
        img_sub = np.subtract(gac_img, doctor_result)
        # True (positive and negative) and False (positive and negative)
        tp = np.float(np.sum((img_sum == 2).nonzero(), dtype=np.int32))
        tn = np.float(np.sum((img_sum == 0).nonzero(), dtype=np.int32))
        fp = np.float(np.sum((img_sub < 0).nonzero(), dtype=np.int32))
        fn = np.float(np.sum((img_sub > 0).nonzero(), dtype=np.int32))
        # Hausdorff equation
        conf_temp = corrcoef(ravel(gac_img), ravel(doctor_result))
        u = directed_hausdorff(doctor_result, gac_img)[0]
        v = directed_hausdorff(gac_img, doctor_result)[0]
        # Other metrics equations
        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        self.dice = (2 * tp) / ((2 * tp) + (fp + fn))
        self.sens = tp / (tp + fn)
        self.spec = tn / (tn + fp)
        self.jacc = tp / (fp + fn + tp)
        self.accu = (tp + tn) / (tp + tn + fp + fn)
        self.f1sc = 2 * ((self.precision * self.recall) /
                         (self.precision + self.recall))
        self.matt = ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) *
                                                   (tn + fp) * (tn + fn))
        self.conf = conf_temp[0, 1]
        self.haus = max(u, v)

    def return_measures(self):
        """
        Return the same measures as FGAC scientific article
            Accuracy (Acc), Matthews Correlation coefficient (MCC),
            Sensitivity (Se), Dice coefficient (DICE), Hausdorff
            distance (HD) and Jaccard index (Jac).
        :return: numpy.array
        """
        from numpy import array
        return array((self.accu, self.matt, self.sens,
                      self.dice, self.haus, self.jacc,
                      self.f1sc, self.spec, self.conf))

    def convert_to_pandas(self, time):
        from pandas import DataFrame
        from numpy import append
        meas = append(self.return_measures(), time)
        df = DataFrame([meas])
        df.columns = [
            'accuracy',
            'matthews',
            'sensitivity',
            'dice',
            'hausdorff',
            'jaccard',
            'f1-score',
            'specificity',
            'conformity',
            'time'
        ]
        return df

    def print_measures(self):
        print(self.df)

    def save_file_measures(self, init):
        file = open("results/file_results/{}.txt".format(init), "w+")
        file.write('\tImage {}.dcm\n\n'.format(init))
        file.write('F-score                 = {}\n'.format(self.f1sc))
        file.write('Accuracy score          = {}\n'.format(self.accu))
        file.write('Dice coefficient        = {}\n'.format(self.dice))
        file.write('Hausdorff distance      = {}\n'.format(self.haus))
        file.write('Jaccard coefficient     = {}\n'.format(self.jacc))
        file.write('Matthews coefficient    = {}\n'.format(self.matt))
        file.write('Conformity coefficient  = {}\n'.format(self.conf))
        file.write('Sensitivity coefficient = {}\n'.format(self.sens))
        file.write('Specificity coefficient = {}\n'.format(self.spec))
        file.close()

    def save_csv(self, img_name):
        pd_file = open("results/pandas_results/{}.csv".format(img_name), "w+")
        self.df.to_csv('results/pandas_results/{}.csv'.format(img_name),
                       header=True,
                       index=True,
                       sep=',',
                       mode='a')
        pd_file.close()
