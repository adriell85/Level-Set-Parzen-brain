"""
morphsnakes
===========
Versão baseada no algoritmo descrito em:
Márquez-Neila, P., Baumela, L., Álvarez, L., "A morphological approach
to curvature-based evolution of curves and surfaces". IEEE Transactions
on Pattern Analysis and Machine Intelligence (PAMI), 2013.
Disponibilizada pelo autor P. Márquez Neila <p.mneila@upm.es>.
"""
from itertools import cycle
import time
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, \
                        gaussian_filter, gaussian_gradient_magnitude
import cv2
#AM: Ajuste no plot do grafico de gradiente
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import subplots_adjust

#FX:
import sys
#sys.path.append('roi')
#sys.path.append('C:/Users/Lapisco03/Desktop/Skin_lesion/roi/') 
from automaticInitialization import return_original_size

class fcycle(object):
    
    def __init__(self, iterable):
        # AM: Cria um iterador circular a partir do parametro em iterable
        # Exemplo: Se iterable = [-1,1], cycle cria uma lista circular -1, 1, -1, 1, -1, 1 ...  
        self.funcs = cycle(iterable)    
    def __call__(self, *args, **kwargs):
        # AM: Retorna o proximo elemento do ciclo
        f = next(self.funcs)
        return f(*args, **kwargs)
    
#####################################################################
# Etapa 1
#####################################################################
# AM: Aqui são projetados os elementos estruturantes pertencentes ao 
# conjunto P de analise (linha de tamanho 3 em 4 orientacoes) 
# ####################################################################    

#AM: Definição dos elementos estruturates dos operadores SI e IS em 2D   
_P2 = [
    # ----------
    #| 1, 0, 0 |
    #| 0, 1, 0 |
    #| 0, 0, 1 |
    # ----------
    np.eye(3), 
    # ----------
    #| 0, 1, 0 |
    #| 0, 1, 0 |
    #| 0, 1, 0 |
    # ----------    
    np.array([[0,1,0]]*3), 
    # ----------
    #| 0, 0, 1 |
    #| 0, 1, 0 |
    #| 1, 0, 0 |
    # ----------    
    np.flipud(np.eye(3)), 
    # ----------
    #| 0, 0, 0 |
    #| 1, 1, 1 |
    #| 0, 0, 0 |
    # ----------    
    np.rot90([[0,1,0]]*3)]

#AM: Definição dos elementos estruturates dos operadores SI e IS em 3D   
_P3 = [np.zeros((3,3,3)) for i in range(9)]
_P3[0][:,:,1] = 1
_P3[1][:,1,:] = 1
_P3[2][1,:,:] = 1
_P3[3][:,[0,1,2],[0,1,2]] = 1
_P3[4][:,[0,1,2],[2,1,0]] = 1
_P3[5][[0,1,2],:,[0,1,2]] = 1
_P3[6][[0,1,2],:,[2,1,0]] = 1
_P3[7][[0,1,2],[0,1,2],:] = 1
_P3[8][[0,1,2],[2,1,0],:] = 1

#AM: Array vazio - dtype=float64
_aux = np.zeros((0))

#####################################################################
# Etapa 2
#####################################################################
# AM: Definicao das funcoes que definem as operacoes morfologicas em SI e IS.
# Aqui eh definido o primeiro componente do operador morfologico de curvatura SI
# Recebe a funcao u, isto e, representacao binaria da superficie do level-set
# e aplica a erosao para coleta do minimo na linha representada pelo EE, depois pega o maximo
#####################################################################
def SI(u):
    """SI operator."""
    global _aux
    #AM: Atribui o elemento estruturante de acordo com a dimensao
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")
    
    #AM: Ajusta a dimensao da forma do levelset
    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)
    
    for _aux_i, P_i in zip(_aux, P):
        # AM: Executa a operacao de erosao binaria entre o E.E e a regiao u
        # AM: Etapa: Infimo / Minimo
        _aux_i[:] = binary_erosion(u, P_i)

    # AM: Etapa: Supremo / Maximo
    return _aux.max(0)

# Aqui eh definido o segundo componente do operador morfologico de curvatura IS
def IS(u):
    """IS operator."""
    global _aux
    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions (should be 2 or 3)")

    #AM: Ajusta a dimensao da forma do levelset
    if u.shape != _aux.shape[1:]:
        _aux = np.zeros((len(P),) + u.shape)
    
    for _aux_i, P_i in zip(_aux, P):
        # AM: Executa a operacao de dilatacao binaria entre o E.E e a regiao u
        # AM: Etapa: Supremo / Maximo
        _aux_i[:] = binary_dilation(u, P_i)

    # AM: Etapa: Infimo / Minimo
    return _aux.min(0)

# SIoIS operator.
# AM: Definicao das funcoes lambda compostas pelos operadores SIoIS
# Define uma lista ciclica de execucao dos operadores em fcycle => [SIoIS, ISoSI, SIoIS, ISoSI, SIoIS, ISoSI...]
SIoIS = lambda u: SI(IS(u))
ISoSI = lambda u: IS(SI(u))
curvop = fcycle([SIoIS, ISoSI])

#####################################################################
# Etapa 3
#####################################################################
# AM: Definicao do criterio de parada baseado no gradiente
# das fronteiras/bordas do objeto com função de suavizacao gaussiana
# Onde:
# - o parametro alpha funciona como um fator de peso para o gradiente
# - o parametro sigma atua definindo a regiao de influencia da funcao
# gaussiana. Este eh o parametro que define o desvio padrao aplicado 
# na gaussiana.
# Stopping factors (function g(I) in the paper).
#####################################################################
def gborders(img, alpha=1.0, sigma=1.0):
    """Stopping criterion for image borders."""    
    # AM: gaussian_gradient_magnitude é uma função de scipy.ndimage.filters para
    # o calculo multidimensional da magnitude do gradiente usando derivadas Gaussianas. 
    # The norm of the gradient.   
    gradnorm = gaussian_gradient_magnitude(img, sigma, mode='constant', cval=0.0)

    #AM: Definicao da funcao g(I) do Contorno ativo geodesico
    return 1.0/np.sqrt(1.0 + alpha*gradnorm)


def gborders2(img):
    """Stopping criterion for image borders."""
    # AM: gaussian_gradient_magnitude é uma função de scipy.ndimage.filters para
    # o calculo multidimensional da magnitude do gradiente usando derivadas Gaussianas.
    # The norm of the gradient.

    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=2)
    erode = cv2.erode(img, kernel, iterations=1)

    result = dilate - erode

    data = result
    data = data/data.max()
    data = data * 255
    data = data.astype(np.uint8)

    return ~data



def glines(img, sigma=1.0):
    """Stopping criterion for image black lines."""
    return gaussian_filter(img, sigma)

# class MorphACWE(object):   
 
#####################################################################
# Etapa 4
#####################################################################
# AM: Definicao do classe do Contorno Ativo Geodesico
# Stopping factors (function g(I) in the paper).
#####################################################################
class MorphGAC(object):
    """Morphological GAC based on the Geodesic Active Contours."""

    #################################################################
    # AM: Parametros do construtor:
    # - data:   Matriz contendo a imagem com aplicacao da funcao gI, 
    #           ou seja, a imagem apos a aplicacao do gradiente com 
    #           derivadas gaussianas.
    # - smoothing: Quantidade de vezes que o operador de curvatura
    #           sera aplicado a cada passo da evolucao do contorno
    #           (simbolo µ)
    # - threshold: Eh o limiar definido (simbolo θ), define os 
    #           limites de atuacao da forca balao, isto e uma vez 
    #           que g(i) retorna valores muito baixos, indicando que 
    #           o contorno esta perto da borda, este limiar atua como 
    #           limite inferior para evitar a atuacao da forca balao
    #           nesta regiao.
    # - balloon:
    #           Define o sentido de evolucao do contorno, uma vez 
    #           que a direcao de evolucao eh normal a curva, positivo
    #           cresce para fora (torna o segmento concavo), negativo
    #           contrai (torna o segmento convexo), padrao 1,-1    
    #################################################################

    def __init__(self, data, smoothing=1, threshold=0, balloon=0):
        """Create a Morphological GAC solver.               
        """
        self._u = None
        self._v = balloon
        self._theta = threshold
        self.smoothing = smoothing
        
        #AM: Recebe a matriz com o g(I) calculada e calcula o gradiente de g(I)
        self.set_data(data)
    
    # AM: A funcao u esta sendo calculada fora    
    def set_levelset(self, u):
        self._u = np.double(u)
        self._u[u>0] = 1
        self._u[u<=0] = 0

    # AM: Atribui o valor do parametro v/balloon e atualiza a mascara
    # precalculada de atuacao da forca balao    
    def set_balloon(self, v):
        self._v = v
        self._update_mask()

    # AM: Ajusta o limiar de atuacao da forca balao    
    def set_threshold(self, theta):
        self._theta = theta
        self._update_mask()

    # AM: Recebe o parametro data = gI e calcula o gradiente da g(I)
    def set_data(self, data):
        self._data = data
        
        # AM: gradiente da funcao ∇gI
        self._ddata = np.gradient(data)
        # AM: pre calcula o mapa de atuacao da forca balao
        self._update_mask()
        # AM: define o elemento estruturante (8-vizinhos) para aplicar 
        # uma dilatacao (v>0) se a curva se expande
        # ou uma erosa (v<0) se a cruva se contrai
        self.structure = np.ones((3,)*np.ndim(data))
    
    #####################
    #AM: Matrix que funciona como mapa para aplicacao da 
    #funcao balao a partir do valor de theta (threshold)
    #cada celula possi o valor 1 ou 0 no indicando que a 
    #forca balao deve ou nao agir     
    def _update_mask(self):
        """Pre-compute masks for speed."""
        self._threshold_mask = self._data > self._theta        
        self._threshold_mask_v = self._data > self._theta/np.abs(self._v)
    
    #AM: Definicao de metodos de acesso e atualizacao dos atributos da classe
    levelset = property(lambda self: self._u,
                        set_levelset,
                        doc="The level set embedding function (u).")
    data = property(lambda self: self._data,
                        set_data,
                        doc="The data that controls the snake evolution (the image or g(I)).")
    balloon = property(lambda self: self._v,
                        set_balloon,
                        doc="The morphological balloon parameter (ν (nu, not v)).")
    threshold = property(lambda self: self._theta,
                        set_threshold,
                        doc="The threshold value (θ).")

    ##########################################################
    #AM: Passo de evolucao do contorno
    ##########################################################
    def step(self):
        """Perform a single step of the morphological snake evolution."""
        
        #AM: define variaveis globais para os parametros
        u = self._u # AM: levelset binario
        gI = self._data # AM: funcao gI
        dgI = self._ddata # AM: gradiente da funcao gI, ∇gI
        theta = self._theta # AM: limiar da forca balao
        v = self._v # AM: define o sentido da forca balao
        
        if u is None:
            raise ValueError("the levelset is not set (use set_levelset)")
        
        #AM: Variavel local para o resultado do passo aplicado sob a regiao u
        res = np.copy(u)
        
        #AM: Define a atuacao da energia balao, 
        # dilata para expandir a curva
        # erode para contrair a curva

        if v > 0:
            aux = binary_dilation(u, self.structure)
        elif v < 0:
            aux = binary_erosion(u, self.structure)

        if v!= 0:
            res[self._threshold_mask_v] = aux[self._threshold_mask_v]
        
        # AM: Ajusta as celulas de res de como uma 
        aux = np.zeros_like(res)
        dres = np.gradient(res)
        for el1, el2 in zip(dgI, dres):
            aux += el1*el2

        # Calcula em aux a soma dos vetores dgi e dres, sendo a soma superior a 0
        # atualiza a regiao interna da curva com valores 1, do contrario atualiza 
        # como 0
        res[aux > 0] = 1
        res[aux < 0] = 0
        
        # Por fim aplica as operacoes morfologicas de curvatura
        for i in range(self.smoothing):
            res = curvop(res)
        
        self._u = res
    
    def run(self, iterations):
        """Run several iterations of the morphological snakes method."""
        for i in range(iterations):            
            self.step()
            #time.sleep(2);
    

def evolve_visual(msnake, original_rows, original_cols, levelset=None, num_iters=20, background=None,count= -1):
    """
    Visual evolution of a morphological snake.    
    """
    from matplotlib import pyplot as plt
    
    if levelset is not None:
        msnake.levelset = levelset
    
    #AM: Captura a imagem atual
    fig = plt.gcf()
    fig.clf()
        
    ###############################################
    #AM: Plota o Gradiente com derivada no subplot
    ###############################################
    ax1 = fig.add_subplot(1,3,1)    
    bar = ax1.imshow(msnake.data, cmap='hot')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(bar, cax=cax)
    subplots_adjust(wspace=0.3)

    ###############################################
    #AM: Plota a curva com background configurado
    # neste caso o background e a imagem de origem
    ###############################################
    ax2 = fig.add_subplot(1,3,2)
    if background is None:
        ax2.imshow(msnake.data, cmap='gray')
    else:
        #ax2.imshow(background, cmap=plt.cm.gray)
        ax2.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))         # Mostra a imagem original colorida

    ax2.contour(msnake.levelset, [0.5], colors='r')
    
    #fig.colorbar(bar, ax=ax2)
    ###############################################
    #AM: Plota a regiao do levelset binario
    ###############################################
    ax3 = fig.add_subplot(1,3,3)
    ax_u = ax3.imshow(msnake.levelset, cmap=plt.cm.gray)

    # AM: Comentada a linha abaixo
    #plt.pause(0.001)
    
    # Iterate.
    for i in range(num_iters):
        # Evolve.
        msnake.step()
        
        # Update figure.
        del ax2.collections[0]
        ax2.contour(msnake.levelset, [0.5], colors='r')
        ax_u.set_data(msnake.levelset)
        fig.canvas.draw()
        plt.pause(0.0001)
        #plt.pause(0.001)
        print("\nIteracao " + str(i))
    # Return the last levelset.

    name = 'C:/Users/Lapisco03/Desktop/Segmentation/PH2-'+ str(count) + '.png'
    #name = 'C:/Users/fabio/Desktop/Segmentation/PH2-'+ str(count) + '.png'
    #name = 'PH2-'+ str(count)
    #name = 'PH2-'+ str(count) + '.png'
    #plt.figure(2)
    #plt.axis('off')
    #plt.imshow(msnake.levelset, cmap=plt.cm.gray)
    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())
    #plt.savefig(name)
    #plt.close()

    #cv2.imshow('Pre Segmentation', msnake.levelset)
    #cv2.waitKey(0)


###################### Pós processamento #########################
    # -> Fechando os espaços dentro do contorno (pode ser bastante melhorada)

    data = msnake.levelset
    data = data/data.max()
    data = data * 255
    data = data.astype(np.uint8)

    #cv2.imshow('Data', data)
    #cv2.waitKey(0)

    #print(data.dtype)

    kernel = np.ones((11,11),np.uint8)
    data = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow(name, msnake.levelset)
    #cv2.waitKey(1)

    #print(msnake.levelset.dtype)

    data = return_original_size(data, original_rows, original_cols)

    cv2.imwrite(name, data)

    return msnake.levelset