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

debug = False

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
    #           o contorno esta perto da borda, este limar atua como 
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

    def adjustDMin(self, x):
        dMin=60
        for a in range(np.size(x,0)):
            p1=False
            p21=False
            p22Pos=0
            p22=False
            p31=False
            p32=False
            p32Pos=0
            p4=False
            d = 0
            for b in range(np.size(x,1)-1):
                #print('>>> 1: ',list(x[a,b:b+2]))
            # print('>>> 2: ', [1,0])
                #print('>>> 3: ', list(x[a,b:b+2]) == [1,0])

                if x[a,b] == 1:
                    p21=True
                elif p21 and x[a,b]==1:
                    p21=False
                elif p21 and x[a,b]==0:
                    p22 = True
                    if p22Pos==0:
                        p22Pos = b
                    d+=1
                elif p22 and x[a,b]==0: 
                    d+=1
                else:
                    p21 = False  

            if d> 0 and d < dMin:
                #print('1: ', p22Pos-(round(d/2)-1))
                #print('2: ', p22Pos+d-2+(round(d/2)-1))
                #print('d: ', d)
                x[a, p22Pos: p22Pos+dMin] = 0
        return x
    
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
            aux += el1*4*el2

        # Calcula em aux a soma dos vetores dgi e dres, sendo a soma superior a 0
        # atualiza a regiao interna da curva com valores 1, do contrario atualiza 
        # como 0
        res[aux > 0] = 1
        res[aux < 0] = 0

        # Por fim aplica as operacoes morfologicas de curvatura
        for i in range(self.smoothing):
            res = curvop(res)
     
        #res = self.adjustDMin(res)
        
        self._u = res
        # cv2.imshow("Res",res)
        
        for x in range(np.size(res,0)):
            line = res[x,:];

    def run(self, iterations):
        """Run several iterations of the morphological snakes method."""
        
        # Iterate.
        analyseStage=10
        areaOld=self._u.sum()
        areaNew=0
        percConvergency=0.00001

        for i in range(iterations):            
            ###############################################
            #AM: Estágio de análise da convergência do contorno
            ###############################################
            if (i+1)%analyseStage==0:                               
                areaNew = self._u.sum()   
                #print('Analyse Stage...')
                #print('- Perc Evolution:', areaNew, areaOld, areaNew/areaOld - 1)
                if areaNew/areaOld - 1 < percConvergency:
                    #print('- Converged!')
                    break
                else:                
                    areaOld = areaNew
            
            self.step()
            #time.sleep(2);
    

def evolve_visual(msnake, levelset=None, num_iters=20, background=None):
    """
    Visual evolution of a morphological snake.    
    """
    from matplotlib import pyplot as ppl
    
    if levelset is not None:
        msnake.levelset = levelset
    
    #AM: Captura a imagem atual
    fig = ppl.gcf()
    fig.clf()

    ###############################################
    #AM: Plota o Contorno Inicial
    ###############################################
    ax0 = fig.add_subplot(2,2,1)
    if background is None:
        ax0.imshow(msnake.data, cmap=ppl.cm.gray)
    else:
        ax0.imshow(background, cmap=ppl.cm.gray)

    ax0.contour(msnake.levelset, [0.5], colors='b')

    ###############################################
    #AM: Plota o Gradiente com derivada no subplot
    ###############################################
    ax1 = fig.add_subplot(2,2,2)    
    bar = ax1.imshow(msnake.data, cmap='hot')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ppl.colorbar(bar, cax=cax)
    subplots_adjust(wspace=0.3)

    ###############################################
    #AM: Plota a curva com background configurado
    # neste caso o background e a imagem de origem
    ###############################################
    ax2 = fig.add_subplot(2,2,3)
    if background is None:
        ax2.imshow(msnake.data, cmap=ppl.cm.gray)
    else:
        ax2.imshow(background, cmap=ppl.cm.gray)

    ax2.contour(msnake.levelset, [0.5], colors='r')
    
    #fig.colorbar(bar, ax=ax2)
    ###############################################
    #AM: Plota a regiao do levelset binario
    ###############################################
    ax3 = fig.add_subplot(2,2,4)
    ax_u = ax3.imshow(msnake.levelset, cmap=ppl.cm.gray)

    # AM: Comentada a linha abaixo
    #ppl.pause(0.001)
    
    # Iterate.
    analyseStage=10
    areaOld=msnake.levelset.sum()
    areaNew=0
    percConvergency=0.00001
    for i in range(num_iters):
        print("\nIteracao "+str(i)) 

        ###############################################
        #AM: Estágio de análise da convergência do contorno
        ###############################################
        if (i+1)%analyseStage==0:                               
            areaNew = msnake.levelset.sum()   
            print('Analyse Stage...')
            print('- Perc Evolution:', areaNew, areaOld, areaNew/areaOld - 1)
            if areaNew/areaOld - 1 < percConvergency:
                print('- Converged!')
                break
            else:                
                areaOld = areaNew
    
        ###############################################
        #AM: Evolução do contorno
        ###############################################
        msnake.step()            
        # Update figure.
        del ax2.collections[0]
        ax2.contour(msnake.levelset, [0.5], colors='r')
        ax_u.set_data(msnake.levelset)
        fig.canvas.draw()
        ppl.pause(10e-3)
    
    return msnake.levelset
