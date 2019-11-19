import morphsnakes

import timeit
import numpy as np
from imageio import imread
from matplotlib import pyplot as ppl

import sys
sys.path.append('roi')
from startPoint import getInitialPoint 
from startPointDoubleLung import getInitialPointLung 
import cv2

#import os
import glob

#----------------------------------------------------
# AM: Conversao da imagem colorida em tons de cinza
#----------------------------------------------------
def rgb2gray(img):    
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

#----------------------------------------------------
# AM: Criacao da funcao inicial do levelset
# Nesta funcao eh construida uma matrix binaria com o circulo de centro (center: cY,cX)
# com raio igual a sqradius, a regiao com 1's define o levelset 0.5 
#----------------------------------------------------
def circle_levelset(shape, center, sqradius, scalerow=1.0):
    #-------------------------------------------------------------------
    # AM: Constrói a função phi do level-set inicial baseado em todo o
    # domínio da imagem (grid: WxH), calculando os pontos internos e externos
    # a função phi inicial
    # 1 - Cria uma malha com as mesmas dimensoes da imagem
    # 2 - Calcula a transposta de (1)
    # 3 - De (2) subtrai o ponto central(cx,cy) de cada coluna: c1-cx, c2-cy
    # 4 - Calcula a transposta do resultado de (3)
    # 5 - De (4) eleva todos os elemento ao quadrado
    # 6 - De (5) soma o elementos da primeira com a segunda matriz, resultando na Matriz Distancia
    # 7 - Calcula a raiz quadrada de (6) 
    # 8 - Subtrai o raio (sqradius) de (7)
    # 9 - Define u como a regiao inicial (Matrix zero levelset) onde os pontos 
    # > 0 em (8) estao dentro da levelset zero e os <= 0 estao fora. A matriz u eh
    # uma matrix binaria, 0 -> fora da regiao, 1 -> dentro da regiao
    #-------------------------------------------------------------------
    print("shape: ", shape);
    data = np.random.randint(5, size=(512, 256))    

    #grid = np.mgrid[list(map(slice, shape))].T - center    
    grid = np.mgrid[list(map(slice, data.shape))].T - center    
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))

    #np.set_printoptions(precision=3)
    #print("phi:\n", phi)    
    u = np.float_(phi > 0)
    #print("u:\n", u)
    return np.hstack((u,u))

def test_GAC(img, p_alpha = 1000, p_auto_ini=True, p_x_ini=0, p_y_ini=0, p_sigma = 6, p_threshold = 0.47,
            p_balloon = 1, p_smoothing = 1, p_num_iters = 70, p_raio = 30, img_name=""):    
    #p_threshold = 0.50 #(Estabilidade prematura)

    #####################################################################
    # AM: Definicao do criterio de parada baseado no gradiente
    # das fronteiras/bordas do objeto com função de suavizacao gaussiana
    # Onde:
    # - o parametro alpha funciona como um fator de peso para o gradiente. 
    # Quanto maior for este valor maior a influencia da magnitude do 
    # gradiente em regioes de diferentes contrastes
    # - o parametro sigma atua definindo a regiao de influencia da funcao
    # gaussiana, ou seja, define o desvio padrao. Quanto maior este valor,  
    # maior sera a regiao de suavizacao da magnitude do gradiente, suavizando
    # o gradiente em regioes onde ha diferentes contrastes. 
    #####################################################################    
    # gI = morphsnakes.gborders(img, alpha=1000, sigma=5.48)
    levelset_zero,grad = getInitialPointLung(img);        
    gI = morphsnakes.gborders(img, alpha=p_alpha, sigma=p_sigma)
    

    ee = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    ee_1 =  np.eye(3)
    ee_2 = np.array([[1.0, 0, 0], [0, 1, 0],[0, 0, 1]])
    ee_3 = np.array([[0, 0, 1], [0, 1, 0],[1, 0, 0]])
    
    ee2 = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    ee3 = np.array([[1, 1],[1, 1], [1, 1],[1, 1],[1, 1]])
    gI = cv2.dilate(gI, ee)    
    
    #####################################################################
    # AM: Fuzzy Frontier Detector - Fuzzy Adjust
    #####################################################################
    
    param_lambda = 0.30
    param_kappa = 0.50
    
    gI[gI<0.33] = 0.10 # g(I) function 
    gI[gI.all()>param_lambda and gI.all()<param_kappa] = 0.34 # g(I) fuzzy
    
    gI = cv2.erode(gI, ee) # local Min    
    gI = cv2.dilate(gI, ee)# local Max
    
    #####################################################################
    # AM: Inicializacao do level-set em toda a dimensão da imagem
    # smoothing: scalar
    #       Corresponde ao numero de repeticoes em que a suavizacao sera
    # aplicada a cada iteracao. Ou seja, a cada passo, serao aplicadas
    # smoothing vezes a funcao de suavizacao. Este procedimento e realizado 
    # na funcao step da classe MorphGAC. Este representa o parametro µ.    
    #
    # threshold : scalar
    #     The threshold that determines which areas are affected
    #     by the morphological balloon. This is the parameter θ.
    # balloon : scalar
    #     The strength of the morphological balloon. This is the parameter ν.
    ##################################################################

    #AM: mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)    
    mgac = morphsnakes.MorphGAC(gI, smoothing=p_smoothing, threshold=p_threshold, balloon=p_balloon)    

    ##################################################################
    # AM: Calcula o ponto de inicialização da curva inicial do level-set
    ##################################################################
    if p_auto_ini :
        iniPoint = getInitialPoint(img_source)    
        iniPointX = iniPoint[0]
        iniPointY = iniPoint[1]
    else:
        iniPointX = p_x_ini
        iniPointY = p_y_ini
    #print('-> iniPoint[0]: '+ str(iniPointX))
    #print('-> iniPoint[1]: '+ str(iniPointY))

    ##################################################################
    # AM: Define a função phi inicial no domínio completo da imagem 
    # (img.shape). Cujo centro é definido pelos pontos (iniPointY, iniPointX)
    # O raio da função phi inicial é definido último parametro, ex: 30.
    # !!TODO!! -> Definir melhor o funcinamento desta função a luz do artigo
    ##################################################################
    mgac.levelset = levelset_zero
    #mgac.levelset, img_norm = getInitialPointLung(img);
    #mgac.levelset = circle_levelset(img.shape, (iniPointY, iniPointX), p_raio)
    
    ##################################################################
    # AM: Visualiza a evolução da curva e inicializa o máximo de interações
    ##################################################################
    ppl.figure()
    morphsnakes.evolve_visual(mgac, num_iters=p_num_iters, background=img)    

    ##################################################################
    # AM: Executa sem visualizar a evoluçao da curva
    ##################################################################
    #mgac.run(iterations=p_num_iters)      
   
    return mgac.levelset;
    
    #cv2.waitKey(0)


#######################
# Principal
#######################
if __name__ == '__main__':
    print("""""")
    cases_name=[]
    
    # Amostras do Coracao
    #img_path = "../../base de imagens/Ecocardiograma/Casos/Todos os Casos"
    #cases_name.append("CasoQD.bmp"); alpha = 1000; num_iters = 100; raio = 30; auto_ini=True; x_ini=0; y_ini=0; sigma = 6; threshold = 0.47; balloon = 1.0; smoothing = 1;

    img_path = "testimages"
    #######################
    # Amostras Pulmao
    #######################
    #cases_name.append("pulmao_7_gray.bmp"); alpha = 1000; num_iters = 100; raio = 30; auto_ini=False; x_ini=150; y_ini=280; sigma = 6; threshold = 0.47; balloon = 1; smoothing = 1;
    #cases_name.append("pulmao_7_gray.bmp"); alpha = 1000; num_iters = 100; raio = 30; auto_ini=False; x_ini=300; y_ini=280; sigma = 6; threshold = 0.47; balloon = 1; smoothing = 1;
    
    #cases_name.append("pulmao_10.bmp"); alpha = 1000; num_iters = 200; raio = 30; auto_ini=False; x_ini=150; y_ini=250; sigma = 6; threshold = 0.47; balloon = 1; smoothing = 1;
    #cases_name.append("pulmao_10.bmp"); alpha = 1000; num_iters = 200; raio = 30; auto_ini=False; x_ini=400; y_ini=250; sigma = 10; threshold = 0.52; balloon = 1; smoothing = 1;

    #cases_name.append("pulmao_12.bmp"); alpha = 1000; num_iters = 200; raio = 30; auto_ini=False; x_ini=150; y_ini=250; sigma = 6; threshold = 0.47; balloon = 1; smoothing = 1;
    
    #cases_name.append("pulmao_29.bmp"); alpha = 1000; num_iters = 150; raio = 20; auto_ini=False; x_ini=130; y_ini=250; sigma = 6; threshold = 0.47; balloon = 1; smoothing = 1;
   
    #****** Casos de testes com pulmao homogeneo ****
    #cases_name.append("pulmao_29_homo.bmp"); alpha = 1000; num_iters = 200; raio = 20; auto_ini=False; x_ini=130; y_ini=250; sigma = 6; threshold = 0.47; balloon = 1; smoothing = 1;
    #cases_name.append("pulmao_29_homo_preto.bmp"); 
    
    ########################################################################
    ## Deu certo para pulmoes
    ########################################################################
    #alpha = 1500; num_iters = 130; raio = 20; auto_ini=False; x_ini=130; y_ini=250; sigma = 6; threshold = 0.54; balloon = 1.0; smoothing = 1;
    #alpha = 1500; num_iters = 130; raio = 20; auto_ini=False; x_ini=130; y_ini=250; sigma = 4; threshold = 0.50; balloon = 1.0; smoothing = 1;
    ########################################################################
    alpha = 1500; num_iters = 130; raio = 20; auto_ini=False; x_ini=130; y_ini=250; sigma = 4; threshold = 0.50; balloon = 1.0; smoothing = 1;

    #best configuration
    #alpha = 1300
    #sigma = 8; threshold = 0.50; balloon = 1.0

    # Amostra Cerebro
    #cases_name.append("cerebro1.jpg"); alpha = 380; num_iters = 50; raio = 10; auto_ini=False; x_ini=150; y_ini=80; sigma = 6; threshold = 0.47; balloon = 1.0; smoothing = 1;
    #cases_name.append("cerebro2.jpg"); alpha = 380; num_iters = 50; raio = 10; auto_ini=False; x_ini=140; y_ini=70; sigma = 6; threshold = 0.47; balloon = 1.0; smoothing = 1;
    #inicio fora
    #cases_name.append("cerebro2.jpg"); alpha = 380; num_iters = 50; raio = 40; auto_ini=False; x_ini=140; y_ini=70; sigma = 6; threshold = 0.47; balloon = -1.0; smoothing = 1;

    # Amostra Melanoma
    #cases_name.append("melanomaIMD022.bmp"); alpha = 1000; num_iters = 100; raio = 40; auto_ini=False; x_ini=250; y_ini=280; sigma = 10; threshold = 0.45; balloon = 1.0; smoothing = 1;
    #cases_name.append("melanomaIMD003.bmp"); alpha = 1000; num_iters = 3; raio = 40; auto_ini=False; x_ini=250; y_ini=280; sigma = 10; threshold = 0.45; balloon = 1.0; smoothing = 1;

    path="testimages/pulmao/*.bmp"
    cases_name = glob.glob(path,recursive=False)
   
    #cases_name.append("13.bmp");
    index=0
    time = [];

    for case_name in cases_name:
        img_source = case_name.replace("\\","/")
        img_name = img_source.rsplit('/',1)[1]
        #img_source = img_path+"/"+case_name          

        #if case_name in ["11.bmp", "13.bmp", "14.bmp", "15.bmp", "17.bmp", "18.bmp", "19.bmp", "8.bmp", "3.bmp", "29.bmp", "30.bmp", "31.bmp", "32.bmp"]:        
        #if case_name in ["35.bmp", "36.bmp", "26.bmp", "27.bmp","28.bmp","29.bmp", "30.bmp", "31.bmp", "32.bmp", "33.bmp"]:                 
        
        # Casos com pulmão esquerdo e direito muito proximos
        #if img_name in ["3.bmp", "8.bmp", "11.bmp"]:
        # Casos com pulmão esquerdo e direito muito proximos
        #if img_name in ["7.bmp", "13.bmp", "14.bmp", "25.bmp", "26.bmp"]:
        # Casos doentes - pulmao irregular
        #if img_name in ["35.bmp", "36.bmp"]:   
        #if img_name in ["11.bmp"]:
        index+=1
            
        print("Index:", index, " - Caso: ", img_source)
        img = imread(img_source)        
        if img.ndim == 3: 
            img = img[...,0]/255.0        
        else:
            img = img/255.0        
        
        #print("img.shape: \n",img.shape)
        #print("img.ndim: \n",img.ndim)

        start = timeit.default_timer()
        getInitialPointLung(img);

        img_number = img_source.rsplit('/',1)[1].split('.',1)[0]
        result = test_GAC(img, p_alpha = alpha, p_auto_ini=auto_ini, p_x_ini=x_ini, p_y_ini=y_ini, p_sigma = sigma, p_threshold = threshold,
                    p_balloon = balloon, p_smoothing = smoothing, p_num_iters = num_iters, p_raio = raio)    
        stop = timeit.default_timer()
        diff = stop - start
        print('Time: ', diff) 
        time.append(diff)

        result = cv2.normalize(result,  result, 0, 255, cv2.NORM_MINMAX)
        #ee2 = np.array([[1, 1, 1], 
        #    [1, 1, 1], 
        #    [1, 1, 1], 
        #    [1, 1, 1], 
        #    [1, 1, 1],                                                                                     
        #    [1, 1, 1]            
        #    ])
        #result = cv2.erode(result, ee2)

        #ee = np.array([[1, 1, 1, 1, 1, 1], 
        #[1, 1, 1, 1, 1, 1], 
        #[1, 1, 1, 1, 1, 1], 
        #[1, 1, 1, 1, 1, 1], 
        #[1, 1, 1, 1, 1, 1],
        #[1, 1, 1, 1, 1, 1]
        #])
        ee = np.array([[1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1]
        ])
        result = cv2.dilate(result, ee)

        cv2.imwrite('D:/Dropbox/2017_Congresso_OPS_Pulmao/Resultados/10/bin/'+img_number+'.png', result)
        cv2.waitKey(0)
        ppl.show()

    print('Time mean: ', np.mean(time), np.std(time))