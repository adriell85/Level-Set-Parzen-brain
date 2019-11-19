def AreaLabel(I):
    """
    Calcular a Area de atraves do bwlabel para uma imagem binaria.
    I - imagem.
    [L,num] = bwlabel(___) also returns num, the number of connected
    objects found in BW. And L returns the label matrix L that contains
    labels for the 8-connected objects found in BW. The label matrix, L,
    is the same size as BW.
    """
    from numpy import sum, sort, argsort, unique
    from cv2 import connectedComponentsWithStats, CV_8U
    output = connectedComponentsWithStats(I, 8, CV_8U)
    l = output[1]  # Label matrix
    area = [sum(l == x) for x in range(0, output[0])]  # Number of labels
    # Return valor, indice and L
    return sort(area), argsort(unique(area)), l
