def get_initial_point_lung(img):
    import numpy as np
    import operator
    import cv2
    import ParzenWindow as pW

    debug = False

    debug_folder = 'results/initialization/'

    if debug:
        cv2.imshow('Original', img)
        cv2.imwrite(debug_folder + '1_original.png', img)

    img_norm = np.zeros_like(img)

    # 1 - Normalization
    cv2.normalize(img, img_norm, 0, 255, cv2.NORM_MINMAX)
    img_norm = cv2.convertScaleAbs(img_norm)

    # 2 - Clip Contrast
    mean, std = cv2.meanStdDev(img_norm)

    ee_3x3 = np.ones((3, 3), dtype=int)
    ee_3x5 = np.ones((3, 5), dtype=int)
    ee_3x7 = np.ones((3, 7), dtype=int)
    ee_5x2 = np.ones((5, 2), dtype=int)
    ee_5x5 = np.ones((5, 5), dtype=int)
    ee_7x3 = np.ones((7, 3), dtype=int)
    ee_7x4 = np.ones((7, 4), dtype=int)
    ee_12x9 = np.ones((12, 9), dtype=int)

    if debug:
        cv2.imshow("1 - image normalized", img_norm)
        cv2.imwrite(debug_folder + '2_img_norm.png', img_norm)

    img_norm[img_norm < mean * 1.00] = 0

    if debug:
        cv2.imshow("2 - clip contrast", img_norm)
        cv2.imwrite(debug_folder + '3_clip_constrast.png', img_norm)

    img_norm = cv2.erode(img_norm, ee_3x3)
    img_norm = cv2.dilate(img_norm, ee_3x3)

    if debug:
        cv2.imshow("2 - clip contrast abertura", img_norm)
        cv2.imwrite(debug_folder + '3_clip_constrast_opening.png', img_norm)

    # 3 - Otsu binarization
    parzen = pW.ParzenWindow(img_norm, lesion=[], background=[])
    img_bin = parzen.segmentation()

    # 4 - Filtering with morphological erosion opening
    img_erode = cv2.erode(img_bin, ee_3x5)
    # FIXME: Talvez ajustar este E.E junte a componente do caso 9
    img_erode = cv2.dilate(img_erode, ee_5x2)

    if debug:
        cv2.imshow('3 - Parzen', img_bin)
        cv2.imwrite(debug_folder + '4_parzen.png', img_bin)
        cv2.imshow('4 - Pos filtro img_bin', img_erode)
        cv2.imwrite(debug_folder + '5_filter_erode_dilate.png', img_erode)

    # 5 - Detecção da maior componente
    connectivity = 4
    output = cv2.connectedComponentsWithStats(img_erode, connectivity, cv2.CV_8U)
    labels = output[1]  # AM: Rotulo das componentes
    stats = output[2]  # AM: Estatistica das componentes
    centroids = output[3]  # AM: Centroids das componentes

    img_max_ = np.zeros(img_bin.shape, img_bin.dtype)
    large_component_1 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    img_max_[labels == large_component_1] = 255
    img_max_[labels != large_component_1] = 0

    img_max_ = cv2.dilate(img_max_, ee_7x4)
    img_max_ = cv2.dilate(img_max_, ee_3x5)

    if debug:
        cv2.imshow("5 - Filtragem Maior Comp.", img_max_)
        cv2.imwrite(debug_folder + '6_largest_component.png', img_max_)

    # 6 - Definição do perimetro baseado no centroide
    # identifica um centroid em img_max_
    ray = 110

    roi_cx = int(centroids[large_component_1, 0])
    roi_cy = int(centroids[large_component_1, 1])
    img_max_inverted = 255 - img_max_
    # Separacao de componentes ligadas, isto evita erro na reconstrucao
    img_max_inverted = cv2.erode(img_max_inverted, ee_3x7)

    img_max_inverted = cv2.erode(img_max_inverted, ee_7x3)

    # AM: FIXME! Raio alterado para extrair imagem para o artigo
    img_roicrop_rect = img_max_inverted[roi_cy - ray:roi_cy + ray, roi_cx - 2 * ray:roi_cx + 2 * ray]

    # corta uma ROI com centro no entroid em img_max_
    if debug:
        cv2.imshow("6 - Definicao do Perimetro", img_roicrop_rect)
        cv2.imwrite(debug_folder + '7_marker_centroid_110ray.png', img_roicrop_rect)

    # 7 - Identificação das duas maiores componentes
    # Identificar as duas maiores componentes
    connectivity = 4
    output = cv2.connectedComponentsWithStats(img_roicrop_rect, connectivity, cv2.CV_8U)
    labels = output[1]
    stats = output[2]

    img_max2_ = np.zeros(img_roicrop_rect.shape, img_roicrop_rect.dtype)
    large_component_1 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    stats[large_component_1, cv2.CC_STAT_AREA] = large_component_1
    largecomponent2 = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()

    # AM: Identifica, com branco, as componentes
    img_max2_[labels == large_component_1] = 255
    img_max2_[labels == largecomponent2] = 255

    img_max_[:, :] = 0
    img_max_[roi_cy - ray:roi_cy + ray, roi_cx - 2 * ray:roi_cx + 2 * ray] = img_max2_

    if debug:
        cv2.imshow("7 - Identificacao das duas componentes", img_max_)
        cv2.imwrite(debug_folder + '8_two_largest_components.png', img_max_)

    # 8 - Reconstrução morfológica das componentes
    img_max_ = img_max_ / 255
    img_max_inverted = img_max_inverted / 255

    # teste linha vertical
    arr_idx_sum = {}
    min_detect = 0
    for y in np.arange(0, np.size(img_max_inverted, 1), 10):
        _sum = np.sum(img_max_inverted[:, y])
        if _sum < 150:
            min_detect += 1
            arr_idx_sum[y] = _sum
    if min_detect > 1:
        sorted_x = sorted(arr_idx_sum.items(), key=operator.itemgetter(1))
        idx_point_b = int(min_detect / 2)
        img_max_inverted[:, min(sorted_x[idx_point_b][0],
                                sorted_x[0][0]): max(sorted_x[idx_point_b][0],
                                                     sorted_x[0][0])] = 0

    # fim do teste linha vertical
    k = 200  # FIXME: Além do k maximo, tentar definir ponto de parada quele quando n houver mudancas
    index = 0
    while index < k:
        img_max_ = cv2.dilate(img_max_, ee_3x3)
        cv2.multiply(img_max_, img_max_inverted, img_max_)
        index = index + 1

    img_max_ = img_max_ * 255

    if debug:
        cv2.imshow("8 - Recontrucao Mask", img_max_inverted)
        cv2.normalize(img_max_inverted, img_max_inverted, 0, 255, cv2.NORM_MINMAX)
        img_max_inverted = cv2.convertScaleAbs(img_max_inverted)
        cv2.imwrite(debug_folder + '9_reconstruction_mask.png', img_max_inverted)
        cv2.imshow("8 - Recontrucao Result", img_max_)
        cv2.imwrite(debug_folder + '9_reconstruction_result.png', img_max_)

    img_max_ = cv2.erode(img_max_, ee_5x5)

    img_max_ = cv2.erode(img_max_, ee_7x3)

    if debug:
        cv2.imshow("Init", img_max_)
        cv2.imwrite(debug_folder + '10_initialization.png', img_max_)
        cv2.waitKey(0)

    img_max_ = cv2.dilate(img_max_, ee_12x9)

    if debug:
        cv2.imshow("Init Dilated", img_max_)
        cv2.imwrite(debug_folder + '11_initialization_dilate.png', img_max_)

    return img_max_, []
