from abc import abstractmethod
import numpy as np
import math
class Gradiente():
    def __init__(self,arr_dz,arr_dw,db):
        self.arr_dz = arr_dz
        self.arr_dw = arr_dw
        self.db = db

    def __str__(self):
        return "db: "+str(self.db)+" arr_dw: "+str(self.arr_dw)

"""
Atividade 1: FUncoes de ativação - altere a resposta das funcoes lambdas correspondentes
de acordo com a especificação
"""

class FuncaoAtivacao():
    def __init__(self,funcao,dz_funcao,dz_ultima_camada=None):
        self.funcao = funcao
        self.dz_funcao = dz_funcao
        self.dz_ultima_camada = dz_ultima_camada

sigmoid = FuncaoAtivacao(lambda z:1/(1+np.power(math.e,-z)),
                      lambda a,z,y,arr_dz_w_prox:np.multiply(a-np.power(a,2), arr_dz_w_prox),
                      lambda a,z,y,arr_dz_w_prox:a-y
                      )

relu = FuncaoAtivacao(lambda z:np.maximum(0,z),
                      lambda a,z,y,arr_dz_w_prox:arr_dz_w_prox*(z>=0)
                      )



leaky_relu = FuncaoAtivacao(lambda z:np.maximum(0.01*z,z),
                      lambda a,z,y,arr_dz_w_prox:arr_dz_w_prox*np.array([1 if  i >= 0 else 0.01 for i in z])
                      )


tanh = FuncaoAtivacao(lambda z:np.tanh(z),
                      lambda a,z,y,arr_dz_w_prox:arr_dz_w_prox*(1-np.power(np.tanh(z), 2))
                      )












class Unidade():
    def __init__(self,func_ativacao,dz_func):

        self.b = 0
        self.func_ativacao = func_ativacao
        self.dz_func = dz_func

        #deixar as linhas de baixo como none, nao foi calculado ainda


        self.arr_w = None
        self.arr_z = None
        self.arr_a = None
        self.mat_a_ant = None
        self.gradiente = None
    def __str__(self):
        return "arr_z: "+str(self.arr_z)+\
                "\narr_a:"+str(self.arr_a)+\
                "\ngradiente: "+str(self.gradiente)+\
                "\nmat_a_ant: "+str(object=self.mat_a_ant)

    def z(self,mat_a_ant):
        return self.arr_w.dot(mat_a_ant.T)+self.b

    def forward_propagation(self,mat_a_ant):
        """
        Função que retorna os resultados da função z por instancia usando a matriz mat_a_ant:
        valores de ativações anterior ou entrada (caso seja primeira camada)
        """
        if(self.arr_w is None):
            self.arr_w = np.random.rand(mat_a_ant.shape[1])*0.01#np.zeros(qtd_pesos)#np.rand(qtd_pesos)
        #print("MAT_A_ANT: "+str(mat_a_ant))
        #print("arr_w: "+str(self.arr_w))
        #print("b: "+str(self.b))

        self.mat_a_ant = mat_a_ant
        self.arr_z = self.z(self.mat_a_ant)
        self.arr_a = self.func_ativacao(self.arr_z)

        #print("ARR_Z: "+str(self.arr_z))
        #print("ARR_A: "+str(self.arr_a))
        return self.arr_a

    def backward_propagation(self,arr_y,arr_dz_w_prox):
        n_instances = len(arr_y)

        #print("arr_a:"+str(self.arr_a)+" arr_z:"+str(self.arr_z)+" arr_y:"+str(arr_y))
        #print("X: "+str(self.mat_a_ant))
        arr_dz = self.dz_func(self.arr_a,self.arr_z,arr_y,arr_dz_w_prox)#self.dz(arr_da)


        #a partir de arr_dz e mat_a_ant, calcula dw considerando todas as instancias
        arr_dw = 1/n_instances * arr_dz.dot(self.mat_a_ant)

        #a partir de arr_dz, calcula db considerando todas as instancias
        db = 1/n_instances * np.sum(arr_dz)

        #print("DZ: "+str(arr_dz))
        #print("arr_dw: "+str(arr_dw))
        #print("db: "+str(db))
        #define o gradiente
        self.gradiente = Gradiente(arr_dz,arr_dw,db)

        return self.gradiente
    def loss_function(self,arr_y):
        return np.sum(-(arr_y*np.log(self.arr_a)+(1-arr_y)*np.log(1-self.arr_a)))/len(arr_y)



    def atualiza_pesos(self,learning_rate):
        self.arr_w = self.arr_w-learning_rate*self.gradiente.arr_dw
        self.b = self.b-learning_rate*self.gradiente.db


class Camada():
    def __init__(self,qtd_unidades,func_ativacao,func_dz):
        self.arr_unidades = []
        self.mat_a = None
        self.ant_camada = None
        self.prox_camada = None
        self.qtd_un_camada_ant = None

        """
        Atividade 2: Inicialize o vetor de arr_unidades com a unidade correspondete
        """
        for i in range(qtd_unidades):
            self.arr_unidades.append(Unidade(func_ativacao,func_dz))



    def forward_propagation(self,mat_a_ant):
        """
        Atividade 3: Calculo da matriz de ativações mat_a a partir da matriz de ativações da camada anterior mat_a_ant.
        Caso seja a primeira camada, mat_a_ant a matriz de entrada do modelo
        """
        #obtenha a quantidade de unidades na camada anterior  por meio de mat_a_ant
        #..pense nas dimensões da matriz
        self.qtd_un_camada_ant = mat_a_ant.shape[1]

        #Inicialize com zeros a matriz de ativacao da camada atual
        #..Novamente, verifique as dimensões da matriz
        self.mat_a = np.zeros((mat_a_ant.shape[0], len(self.arr_unidades)))

        #print("MAT A:"+str(self.mat_a.shape))
        #print("MAT_A_ANT: "+str(mat_a_ant))
        #para cada unidade, realiza o forward propagation
        #...o forward_propagation da unidade retorna um vetor arr_a com as ativações
        #... por instancia. Você deve armazenar os valores corretamente na matriz mat_a (veja especificação)
        for i,unidade in enumerate(self.arr_unidades):
            arr_a = unidade.forward_propagation(mat_a_ant) 
            self.mat_a[:,i] = arr_a

        return self.mat_a

    #atividade 4: Complete as propriedades mat_w, mat_dz e mat_dz_w
    @property
    def mat_w(self):
        """
        Cria a matriz _mat_w a partir dos vetores de pesos de cada unidade
        Verifique as dimensões da matriz (preencha os None)
        """
        #inicialize com zero a matriz
        #de acordo com as suas dimensoes
        _mat_w = np.zeros((len(self.arr_unidades), self.qtd_un_camada_ant))

        #para cada unidade, preencha corretamente os valores da matriz
        #usando o vetor de pesos de cada unidade
        for i,unidade in enumerate(self.arr_unidades):
            _mat_w[i,:] = unidade.arr_w

        return _mat_w

    @property
    def mat_dz(self):
        """
        Cria a matriz _mat_dz a partir dos vetores arr_dz do gradiente de cada unidade
        Verifique as dimensões da matriz (preencha os None)
        """
        #inicialize com zero a matriz
        _mat_dz = np.zeros((self.mat_a.shape[0], len(self.arr_unidades)))

        #para cada unidade, preencha corretamente os valores da matriz
        for i,unidade in enumerate(self.arr_unidades):
            _mat_dz[:,i] = unidade.gradiente.arr_dz

        return _mat_dz

    @property
    def mat_dz_w(self):
        """
        Realiza o calculo do produto entre mat_dz e mat_w
        chame as propriedades correspondentes
        """
        return np.matmul(self.mat_dz,self.mat_w)

    def backward_propagation(self,arr_y):
        """
        Atividade 5: Realiza o calculo do backward propagation
        """
        #obtenha o mat_dz_w da proxima camada
        #..Caso não exista proxima camada, mat_dz_w_prox permanecerá None
        mat_dz_w_prox = self.prox_camada.mat_dz_w if self.prox_camada is not None else None

        for i,unidade in enumerate(self.arr_unidades):
            #Caso exista mat_dz_w_prox, obtenha o arr_dz_w_prox
            #..correspondente a esta unidade. Para isso, fique atento a dimensão de mat_dz_w_prox
            arr_dz_w_prox =  mat_dz_w_prox[:,i] if mat_dz_w_prox is not None else None

            #chame o backwrd_propagation desta unidade
            unidade.backward_propagation(arr_y, arr_dz_w_prox)

    def atualiza_pesos(self,learning_rate):
        """
        Já está pronto: para cada unidade, atualiza seus pesos
        """
        for unidade in self.arr_unidades:
            unidade.atualiza_pesos(learning_rate)



class RedeNeural():
    def __init__(self,arr_qtd_un_por_camada,
                    arr_func_a_por_camada,
                    num_iteracoes):
        #Vetor que armazena a lista de instancias da classe Camada dessa Rede Neural. Esses objetos são instanciados por meio do método config_rede
        self.arr_camadas = []
        #Vetor de inteiros que, o valor na posição i do vetor corresponde a quantidade de unidades na i-ésima camada.
        self.arr_qtd_un_por_camada = arr_qtd_un_por_camada
        #Função de ativação por camada. Esse vetor armazena objetos da classe FuncaoAtivacao em que, cada posição i do vetor, corresponde a função de ativação da i-ésima camada
        self.arr_func_a_por_camada = arr_func_a_por_camada
        #Número de iterações (épocas) que a rede neural irá rodar
        self.num_iteracoes = num_iteracoes
        #Representa o vetor y. Esses objetos são instanciados por meio do método config_rede
        self.arr_y = []
        #Representa a matriz de atributos por instancias X. Esses objetos são instanciados por meio do método config_rede
        self.mat_x = None

    def config_rede(self,mat_x,arr_y):
        """
        Atividade 6 - Configura a rede: armazena em arr_camada todas as camadas por meio da
        quantidade de unidades por camada e funções de ativação correspondente, inicializa mat_x e arr_y
        """
        self.mat_x = mat_x
        self.arr_y  = arr_y

        #para cada camada, ao i
        for camada_l,qtd_unidades in enumerate(self.arr_qtd_un_por_camada):
            #por meio de arr_func_a_por_camada defina a dz_função que será usada
            #..caso seja a ultima camada, será usada a dz_ultima camada
            dz_ultima_camada = self.arr_func_a_por_camada[camada_l].dz_ultima_camada
            dz_funcao = self.arr_func_a_por_camada[camada_l].dz_funcao if(camada_l<len(self.arr_qtd_un_por_camada)-1) else dz_ultima_camada
            #instancie a camda
            obj_camada = Camada(qtd_unidades, self.arr_func_a_por_camada[camada_l].funcao, dz_funcao)
            self.arr_camadas.append(obj_camada)
            #armazena a camada anterior
            if(camada_l>0):
                obj_camada.ant_camada = self.arr_camadas[camada_l-1]

        #para cada camada até a penultima, armazene em camada.prox_camada a camada seguinte
        for l,camada in enumerate(self.arr_camadas):
            if(l<len(self.arr_camadas)-1):
                camada.prox_camada = self.arr_camadas[l+1]

    def forward_propagation(self):
        """
        Atividade 7: Execute, para todas as camadas, o método forward_propagation.
        """
        num_camadas = len(self.arr_camadas)
        mat_a = self.arr_camadas[0].forward_propagation(self.mat_x)
        i=1
        while i < num_camadas:
            mat_a = self.arr_camadas[i].forward_propagation(mat_a)
            if i == num_camadas:
                break
            i+=1
        return mat_a
            


    def backward_propagation(self):
        """
        Atividade 8: Execute, para todas as camadas, o método backward_propagation. Fique atento na ordem de execução
        """
        num_camadas = len(self.arr_camadas)
        for i in range(num_camadas-1,-1,-1):
            self.arr_camadas[i].backward_propagation(self.arr_y)


    def atualiza_pesos(self,learning_rate):
        """
        Já está pronto: chama o atualiza pesos das camadas
        """
        for camada in self.arr_camadas:
            camada.atualiza_pesos(learning_rate)
        #print("arr_w: "+str(self.arr_camadas[0].arr_unidades[0].arr_w))
        #print("gradiente: "+str(self.arr_camadas[0].arr_unidades[0].gradiente))

    def fit(self,mat_x,arr_y,learning_rate=1.1):
        """
        Atividade 9: Realiza self.num_iteracoes iterações (épocas) da rede neural
        """
        #primeiramente, você deve configurar a rede
        self.config_rede(mat_x,arr_y)

        for i in range(self.num_iteracoes):
            #faça aqui a execução desta iteração
            self.mat_x = mat_x
            self.arr_y = arr_y
            self.forward_propagation()
            self.backward_propagation()
            self.atualiza_pesos(learning_rate)
            
            #print("A: "+str(self.arr_camadas[0].arr_unidades[0].arr_a))
            #print("Y:"+str(arr_y))
            if(i % 100 == 0):
                loss = self.loss_function(arr_y)
                print("Iteração: "+str(i)+" Loss: "+str(loss))



    def loss_function(self,arr_y):
        """
        Atividade 10: Calcula a função de perda por meio do vetor de ativações
        """
        #Para calcular o loss_function, voce deverá obter o vetor de
        #..ativações (arr_a) apropriado. Fique atento com qual camada/unidade você deverá
        #..obter o arr_a. Preencha os None com o valor apropriado

        arr_a = self.arr_camadas[0].arr_unidades[0].arr_a
        #print("ARRAY Y: "+str(arr_y))
        #print("ARRAY A: "+str(arr_a))

        return np.sum(-(arr_y*np.log(arr_a)+(1-arr_y)*np.log(1-arr_a)))/len(arr_y)

    def predict(self,mat_x):
        """
        Atividade 11: faz a predição, para uma matriz de instancias/atributos mat_x
        """
        #faz a predição, para uma matriz de instancias/atributos mat_x
        #..para realizar a predição, você deverá fazer o mesmo que na regressão logistica
        #..porém, deverá ficar atento a qual camada/unidade você deverá obter o vetor de ativações arr_a
        #..preencha os none com o valor apropriado
        self.mat_x = mat_x
        mat_a = self.forward_propagation()

        #print(self.arr_a)
        arr_a = self.arr_camadas[0].arr_unidades[0].arr_a

        return (1*(mat_a>0.5))
