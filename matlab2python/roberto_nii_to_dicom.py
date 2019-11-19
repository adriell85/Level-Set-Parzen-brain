import cv2
import os
import natsort
# from framework import load_nii
import numpy as np
import cv2


def normalize_image(v):
    v = (v - v.min()) / (v.max() - v.min())
    result = (v * 255).astype(np.uint8)
    return result


def main(path_nii, path_gt, folder_original, folder_gt):
    lst_files = []
    lst_files_gt = []

    for dir_gt, subdir_list_gt, file_list_gt in os.walk(path_gt):
        file_list_gt = natsort.natsorted(file_list_gt, reverse=False)

        for filename in file_list_gt:
            if ".nii.gz" in filename.lower():
                lst_files_gt.append(os.path.join(path_gt, filename))

    for dir_name, subdir_list, file_list in os.walk(path_nii):
        file_list = natsort.natsorted(file_list, reverse=False)

        for filename in file_list:
            if ".nii.gz" in filename.lower():
                lst_files.append(os.path.join(path_nii, filename))

    for index_path in range(len(lst_files)):
        volume, _, _ = cv2.imread(lst_files[index_path])
        volume_gt, _, header = cv2.imread(lst_files_gt[index_path])

        x, y, z = volume.shape
        print('Exame: {}'.format(index_path))

        # total_images = 1

        for k in range(z):
            img = normalize_image(volume[:, :, k])
            gt = normalize_image(volume_gt[:, :, k])

            cv2.imwrite(os.path.join(folder_original, '{0}/Imagem{1}.png'.format(index_path, k)), img)
            cv2.imwrite(os.path.join(folder_gt, '{0}/Imagem{1}.png'.format(index_path, k)), gt)
            # total_images += 1
            # vis = np.concatenate((img, gt), axis=1)
            # cv2.imshow('ly', vis)
            # cv2.waitKey(0)


if __name__ == "_main_":
    path_nii = '/home/meupc/Documents/LAPISCO/Datasets/HVSMR2016/Training dataset/short_axis/Training dataset/nii/'
    path_gt = '/home/meupc/Documents/LAPISCO/Datasets/HVSMR2016/Training dataset/short_axis/Ground truth/nii/'
    path_folder_original = '/home/meupc/Documents/LAPISCO/Datasets/HVSMR2016/Training dataset/short_axis/Training dataset/png/'
    path_folder_gt = '/home/meupc/Documents/LAPISCO/Datasets/HVSMR2016/Training dataset/short_axis/Ground truth/png/'

    main(path_nii, path_gt, path_folder_original, path_folder_gt)