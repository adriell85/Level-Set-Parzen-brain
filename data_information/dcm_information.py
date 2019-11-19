# -*- coding: utf-8 -*-


def show_figures(image, number):
    """
    Show figures without borders and white spaces in plot
    :param image: 2d array
    :param number: int
    :return: int
    """
    import matplotlib.pyplot as ppl
    ppl.figure(number)
    ppl.imshow(image, cmap='gray')
    return number + 1


def m_uint8(img):
    from cv2 import convertScaleAbs
    img[img < 0] = 0
    return convertScaleAbs(img)


def load_dcm(img_path):
    """
    Load dcm images as numpy array
    :param img_path: string
    :return: np.array
    """
    from pydicom import read_file
    from numpy import array, int16
    return array(read_file(img_path).pixel_array, dtype=int16)


def get_database(filename, indent=0):
    """
    Go through all items in the dataset and print them with custom format
    Modelled after Dataset._pretty_str()
    """
    import pydicom
    import matplotlib.pyplot as plt

    dont_print = ['Pixel Data', 'File Meta Information Version']

    indent_string = "   " * indent
    next_indent_string = "   " * (indent + 1)

    for data_element in filename:
        if data_element.VR == "SQ":  # a sequence
            print(indent_string, data_element.name)
            for sequence_item in data_element.value:
                get_database(sequence_item, indent + 1)
                print(next_indent_string + "---------")
        else:
            if data_element.name in dont_print:
                print("""<item not printed -- in the "don't print" list>""")
            else:
                repr_value = repr(data_element.value)
                if len(repr_value) > 50:
                    repr_value = repr_value[:50] + "..."
                print("{0:s} {1:s} = {2:s}".format(indent_string,
                                                   data_element.name,
                                                   repr_value))

    # 2 ^ 11 bytes = 2048
    filename = "datasets/ImagensTC_Pulmao/8.dcm"
    dataset = pydicom.dcmread(filename, force=True)
    dataset.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    # dataset.pixel_array

    # filename = get_testdata_files('MR_small.dcm')[0]
    # ds = pydicom.dcmread(filename)
    # myprint(ds)

    # Normal mode:
    print()
    print("Filename.........:", filename)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", dataset.PatientID)
    print("Modality.........:", dataset.Modality)
    print("Study Date.......:", dataset.StudyDate)
    # print("Bit Depth........:", dataset.pixel_array)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(rows=rows,
                                                                              cols=cols,
                                                                              size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)

    # use .get() if not sure the item exists, and want a default value if missing
    print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

    get_database(dataset)

    # plot the image using matplotlib
    plt.imshow(dataset.pixel_array, cmap=plt.bone())
    plt.show()
