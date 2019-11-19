#!/usr/bin/env python
#########################################
#       nii2png for Python 3.7          #
#         NIfTI Image Converter         #
#                v0.2.9                 #
#                                       #
#     Written by Alexander Laurence     #
# http://Celestial.Tokyo/~AlexLaurence/ #
#    alexander.adamlaurence@gmail.com   #
#              09 May 2019              #
#              MIT License              #
#########################################

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



def main(argv):
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
              for filename in val_enter:
                  if 'nii' in filename.lower():
                    input = direct+'/'+folder+'/'+arq+'/'+filename
                    output= 'C:/Users/gomes/Desktop/training translate/'+folder+'/'+arq
                      # print(output)
                      # print(input)


                    inputfile = input
                    outputfile = output

                    try:
                        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
                    except getopt.GetoptError:
                        print('nii2png.py -i <inputfile> -o <outputfile>')
                        sys.exit(2)
                    for opt, arg in opts:
                        if opt == '-h':
                            print('nii2png.py -i <inputfile> -o <outputfile>')
                            sys.exit()
                        elif opt in ("-i", "--input"):
                            inputfile = arg
                        elif opt in ("-o", "--output"):
                            outputfile = arg

                    print('Input file is ', inputfile)
                    print('Output folder is ', outputfile)

                    # set fn as your 4d nifti file
                    image_array = nibabel.load(inputfile).get_data()
                    print(len(image_array.shape))

                    # ask if rotate
                    ask_rotate = 'y'

                    if ask_rotate == 'y':
                        ask_rotate_num = 90
                        if ask_rotate_num == 90 or ask_rotate_num == 180 or ask_rotate_num == 270:
                            print('Got it. Your images will be rotated by {} degrees.'.format(ask_rotate_num))
                        else:
                            print('You must enter a value that is either 90, 180, or 270. Quitting...')
                            sys.exit()
                    elif ask_rotate.lower() == 'n':
                        print('OK, Your images will be converted it as it is.')
                    else:
                        print('You must choose either y or n. Quitting...')
                        sys.exit()

                    # if 4D image inputted
                    if len(image_array.shape) == 4:
                        # set 4d array dimension values
                        nx, ny, nz, nw = image_array.shape

                        # set destination folder
                        if not os.path.exists(outputfile):
                            os.makedirs(outputfile)
                            print("Created ouput directory: " + outputfile)

                        print('Reading NIfTI file...')

                        total_volumes = image_array.shape[3]
                        total_slices = image_array.shape[2]

                        # iterate through volumes
                        for current_volume in range(0, total_volumes):
                            slice_counter = 0
                            # iterate through slices
                            for current_slice in range(0, total_slices):
                                if (slice_counter % 1) == 0:
                                    # rotate or no rotate
                                    if ask_rotate == 'y':
                                        if ask_rotate_num == 90 or ask_rotate_num == 180 or ask_rotate_num == 270:
                                            print('Rotating image...')
                                            if ask_rotate_num == 90:
                                                data = numpy.rot90(image_array[:, :, current_slice, current_volume])
                                            elif ask_rotate_num == 180:
                                                data = numpy.rot90(numpy.rot90(image_array[:, :, current_slice, current_volume]))
                                            elif ask_rotate_num == 270:
                                                data = numpy.rot90(
                                                    numpy.rot90(numpy.rot90(image_array[:, :, current_slice, current_volume])))
                                    elif ask_rotate.lower() == 'n':
                                        data = image_array[:, :, current_slice, current_volume]

                                    # alternate slices and save as png
                                    print('Saving image...')
                                    image_name = inputfile[:-4] + "_t" + "{:0>3}".format(
                                        str(current_volume + 1)) + "_z" + "{:0>3}".format(str(current_slice + 1)) + ".png"

                                    # scipy.misc.imsave(image_name, data)]
                                    cv2.imwrite(image_name,normalizeImage(data))
                                    # cv2.imshow(image_name,normalizeImage(data))
                                    # cv2.waitKey(0)
                                    # count_debug = di.show_figures(data, 1)
                                    # ppl.show()

                                    print('Saved.')

                                    # move images to folder
                                    print('Moving files...')
                                    src = image_name
                                    shutil.move(src, outputfile)
                                    slice_counter += 1
                                    print('Moved.')

                        print('Finished converting images')

                    # else if 3D image inputted
                    elif len(image_array.shape) == 3:
                        # set 4d array dimension values
                        nx, ny, nz = image_array.shape

                        # set destination folder
                        if not os.path.exists(outputfile):
                            os.makedirs(outputfile)
                            print("Created ouput directory: " + outputfile)

                        print('Reading NIfTI file...')

                        total_slices = image_array.shape[2]

                        slice_counter = 0
                        # iterate through slices
                        for current_slice in range(0, total_slices):
                            # alternate slices
                            if (slice_counter % 1) == 0:
                                # rotate or no rotate
                                if ask_rotate.lower() == 'y':
                                    if ask_rotate_num == 90 or ask_rotate_num == 180 or ask_rotate_num == 270:
                                        if ask_rotate_num == 90:
                                            data = numpy.rot90(image_array[:, :, current_slice])
                                        elif ask_rotate_num == 180:
                                            data = numpy.rot90(numpy.rot90(image_array[:, :, current_slice]))
                                        elif ask_rotate_num == 270:
                                            data = numpy.rot90(numpy.rot90(numpy.rot90(image_array[:, :, current_slice])))
                                elif ask_rotate.lower() == 'n':
                                    data = image_array[:, :, current_slice]

                                # alternate slices and save as png
                                if (slice_counter % 1) == 0:
                                    print('Saving image...')
                                    image_name = inputfile[:-4] + "_z" + "{:0>3}".format(str(current_slice + 1)) + ".png"

                                    # scipy.misc.imsave(image_name, data)
                                    cv2.imwrite(image_name,normalizeImage(data))
                                    # cv2.imshow(image_name, normalizeImage(data))
                                    # cv2.waitKey(0)
                                    # count_debug = di.show_figures(data, 1)
                                    # ppl.show()
                                    print('Saved.')

                                    # move images to folder
                                    print('Moving image...')
                                    src = image_name
                                    # shutil.move(src, outputfile)
                                    slice_counter += 1
                                    print('Moved.')

                        print('Finished converting images')
                    else:
                        print('Not a 3D or 4D Image. Please try again.')


# call the function to start the program
if __name__ == "__main__":
    main(sys.argv[1:])