def withinSkull(imgOrig, flag):
    """
    fuzzy c-mean image segmentation with weighted
    :param imgOrig: grayscale image
    :param flag:
    :param count_debug:
    :return: imgUtil: 2d array
        Multiplicao de imgOrig e skull,
    :return: skullInside: 2d array
        Parte interior do cerebro
    :return: skull: 2d array
        Cerebro sendo 1, resto sendo 0
    :return: se: 2d array
        Elementro estruturante 19x19
    """
    from data_information.dcm_information import m_uint8
    from os.path import abspath, join, dirname
    import numpy as np
    import cv2
    import sys

    sys.path.insert(0, abspath(join(dirname(__file__), '..')))
    from data_information import dcm_information as di

    from AreaLabel import AreaLabel

    int1 = np.uint8(1)
    int0 = np.uint8(0)
    b_matrix = np.where(imgOrig >= 255, int1, int0)

    # Fechamento
    # Array of 19x19
    ee_str = np.array((
        [[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
    ), dtype=np.uint8)

    for x in range(0, 9):
        b_matrix = cv2.erode(cv2.dilate(b_matrix, ee_str), ee_str)

    _, index, L = AreaLabel(b_matrix)

    # Calcular a Area de todos os label encontrados.
    if len(index) >= 1:
        skull = np.where(L == index[0], int0, int1)
    elif len(index) == 0:
        skull = 1 - L

    # [Matheus] Codigo para representar 'floodfill' do MATLAB
    im_flood_fill = np.copy(skull)
    h, w = skull.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    # Invert floodfilled image
    # Combine the two images to get the foreground
    skull_inside = np.bitwise_or(skull, cv2.bitwise_not(im_flood_fill))

    # Segunda Etapa
    # Realizar uma erosao
    # Obs: realizar uma erosao umas 5 ou 6 vezes para retirar as bordas.
    ee_str_2 = np.ones((5, 5), dtype=np.uint8)  # ElementoEstruturante 15
    if flag == 1:
        for i in range(0, 3):  # 10
            skull_inside = cv2.erode(1 - skull, ee_str_2)

        valor, index, L = AreaLabel(skull_inside)

        if len(index) > 1:
            skull_inside = np.where(L == index[2], int1, int0)
        elif len(index) == 0:
            skull_inside = L.copy()
        else:
            skull_inside = np.where((1 - skull) == index,
                                    int1, int0)
    # img_util, skull_inside, skull, ee_str
    return m_uint8(imgOrig) * skull_inside, \
           skull_inside, skull, ee_str, (h, w)
