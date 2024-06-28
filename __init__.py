import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

sep = r'/'

base_treino = 'src/base_perceptron_desbalanceada_treino.csv'
base_teste = 'src/base_perceptron_balanceada_teste.csv'
BIAS = 0.5


def create_database(n:int=2, balanced:bool=True) -> None:
    """
    Cria bases de dados para implementação do perceptron.
    
    --------parâmetros---------
    n: número inteiro de classes desejadas, padronizado em 2.
    balanced: variável booleana. True se as classes devem ser balanceadas,
              False caso contrário.

    O nome de salvamento da base no final deve ser trocado conforme a necessidade,
    especificando se a base é de treino ou teste.
    As variaveis x_mean, y_mean alteram o ponto da média dos agrupamentos.
    rad_x e rad_y alteram o formato da distribuição (gaussiana)
    """
    x_mean = -1
    y_mean = -1

    rad_x = 2
    rad_y = 2

    df = pd.DataFrame()
    for i in range(n):
        aux = pd.DataFrame(columns=['X', 'Y', 'CLASSE'])
        if i!=0:
            soma = 8

            x_mean += soma
            y_mean += 1.5*soma

        mean = [x_mean, y_mean]
        cov = [[rad_x, 0], [0, rad_y]]

        num = 20

        if balanced==False:
            if i%2!=0:
                num = 7
            x, y = np.random.multivariate_normal(mean, cov, num).T
        else:
            x, y = np.random.multivariate_normal(mean, cov, num).T
        
        plt.plot(x, y, 'x')

        aux['X'] = x
        aux['Y'] = y
        aux['CLASSE'] = i

        df = pd.concat([df,aux], ignore_index=True)

    if balanced == True: name = 'balanceada'
    else: name = 'desbalanceada'


    df.to_csv(os.getcwd()+sep+'src'+sep+'base_perceptron_'+ name+'_treino.csv', index=False)

    plt.axis('equal')
    plt.grid()
    plt.show()

    return 

def activation_func(v:float):
    """
    função de ativação do perceptron.
    -----------PARÂMETROS-------------
    v: (float) resultado do calculo do perceptron (w*x + bias)
    ----------RETORNO----------------
    classe do objeto (int).
    """
    return 1 if v>=BIAS else 0

def perceptron(X, w,bias=BIAS):
    """
    Função de utilização do perceptron treinado.
    --------PARÂMETROS-------------
    X: (array) vetor de entradas dos objetos que se deseja classificar
    w: (array) vetor de pesos obtido do treinamento do perceptron
    -------RETORNO--------------------
    resultado de classificação de cada objeto presente em X (array).
    """
    result = []
    for entrada in X:
        y = activation_func(np.matmul(w,entrada)+BIAS)
        result.append(y)
    result = np.array(result)
    return result

def obtain_folds(base,nfolds):
    """
    Obtem os indices dos elementos de cada pasta para a validação cruzada.
    ----------PARÂMETROS----------
    base: string com o caminho da base de treino, utilizada na validação. 
    nfolds: número de pastas utilizadas na validação cruzada.
    -----------RETORNO-----------
    aux: (array) indices dos elementos de cada classe da base de dados.
    """
    df = pd.read_csv(base)
    df = df.sample(frac=1,random_state=42).reset_index(drop=True)

    n = df['CLASSE'].value_counts()
    
    n_min = n.min() #quantidade minima de elementos possiveis para cada classe
    
    assert n_min >= nfolds,f'Uma das classes não possui elementos suficientes para a divisão em {nfolds}'

    # quantidade de objetos de cada classe por pasta
    n_unique_objects_fold = np.array([np.floor(i/n_min) for i in n])
    
    index_list = []
    classes = np.array(df['CLASSE'].unique())

    #indices dos objetos de cada classe dentro da base de dados
    for classe in classes:
        index_list.append(np.where(df['CLASSE']==classe))
    
    index_list = [i[0] for i in index_list]

    element_index_list = []

    j = 0
    for lista in index_list:
        #separa o grupos de indices para cada pasta com base no numero de objetos de cada classe calculado anteriormente
        indices_separados = [lista[i*int(n_unique_objects_fold[j]):(i+1)*int(n_unique_objects_fold[j])] for i in range(nfolds)]
        j=j+1
        element_index_list.append(indices_separados)

    #junta os indices dos elementos de cada classe
    aux = []
    for i in range(len(element_index_list[0])):
        classe1 = element_index_list[0][i]
        classe2 = element_index_list[1][i]
        aux.append(np.concatenate((classe1,classe2)))

    return aux

def split(skf,nfolds):
    """
    Separa a base de dados de treino em pastas para a validação cruzada
    -------PARÂMETROS--------
    skf: (array) indices dos elementos de cada classe na base de dados.
    nfolds: (int) numero de pastas para a validação cruzada.
    ----------RETORNO--------
    train_indexes: (array) indices na base de dados dos elementos de treino de treino para cada etapa da validação cruzada.
    test_indexes: (array) indices na base de dados dos elementos de teste para cada etapa da validação cruzada.
    """
    train_indexes = []
    test_indexes = []
    for i in range(nfolds):
        aux = []
        for j in range(nfolds):
            if j!=i:
                aux.append(skf[j])
            else:
                test_indexes.append(skf[i])
        aux = [i for sublista in aux for i in sublista]
        train_indexes.append([int(i) for i in aux])

    return train_indexes,test_indexes
    

def cross_validation(base:str, nfolds=10):
    """
    Implementa validação cruzada para bases de dados estratificadas,
    utilizando o StratifiedKFold para a construção da base.
    Numero de dobras padronizado em 10.
    ------PARÂMETROS------
    base: (string) caminho para a base primária para gerar as bases da validação cruzada.
    nfolds: (int) número de pastas para a divisão da base de dados para a validaçao cruzada.
    -----RETORNO----------
    MSE: (float array) erros médios quadráticos de cada etapa de validação
    std: (float array) desvios padrão de cada dos erros quadráticos de cada etapa de validação.
    """

    df = pd.read_csv(base, sep=',', decimal='.')

    df = df.sample(frac=1,random_state=42).reset_index(drop=True)

    skf = obtain_folds(base,nfolds)
    
    indices_treino, indices_teste = split(skf,nfolds)

    MSE = np.ones(nfolds)
    stds = np.ones(nfolds)
    folds = np.arange(0,nfolds,1)
    for i,train_index, test_index in zip(folds,indices_treino,indices_teste):
        X_treino = np.array(df.drop(columns=['CLASSE']).iloc[train_index])
        Y_treino = np.array(df.drop(columns=['X','Y']).iloc[train_index])
        w, e = train_model(X_treino, Y_treino, bias=BIAS,learn_rate=0.1,nepocas=10)
        
        pesos_teste = w
        
        X_teste = np.array(df.drop(columns=['CLASSE']).iloc[test_index])
        Y_teste = np.array(df.drop(columns=['X','Y']).iloc[test_index])

        resultados = perceptron(X_teste, pesos_teste, BIAS)

        # calculo do erro quadratico medio
        erro_quad = 0
        for j in range(len(resultados)):
            
            erro_quad = erro_quad+ (Y_teste[j][0] - resultados[j])**2

        MSE[i] = erro_quad/len(resultados)
        stds[i] = np.std(erro_quad)
        

    return MSE,stds

def train_model(X:np.array, Y:np.array, bias:float=0, learn_rate:float=0.01,nepocas:int=10,verbose=False):
    """
    Realiza o treinamento do perceptron.
    -------PARÂMETROS------
    X: (array dimensão n x 2) contém os valores de variáveis usadas na classificação
    Y: (array dimensão n) contém as classes de cada objeto.
    learn_rate: (float) taxa de aprendizado do perceptron. Valor padrão 0.01
    bias: (float) bias do perceptron. Valor padrão 0
    -------RETORNO----------
    w: (array) vetor de pesos ajustado após o treino do modelo.
    erro_med: (float) erros quadráticos médios a cada época de treinamento.
    """
    lin,col = np.shape(X)
    w = np.full(col, 0)
    erro_med = []
    
    for i in range(nepocas):   
        erro = 0 
        for entrada, objetivo in zip(X, Y):
            y = activation_func(np.matmul(w,entrada)+bias)
            erro = erro+(objetivo - y)**2
            w = w + learn_rate*(objetivo - y)* entrada

        erro_med.append(erro/len(Y))
    erro_med = np.array(erro_med)
    plot_fronteira_decisao(X,Y,w,bias) if verbose==True else 0

    return w, erro_med

def fronteira_decisao(x, weight, bias=BIAS):
        """
        Calcula a fronteira de decisão de um perceptron treinado para um conjunto de pontos.
        Função aplicável para problemas de classificação binária em 2D.
        ---------PARÂMETROS------------
        x: (array) pontos nos quais se deseja calcular a fronteira de decisão.
        weight: (array) vetor de pesos do perceptron treinado.
        bias: (float, optional) bias utilizado durante o treinamento.
        -----------RETORNO-------------
        valores (x,y) de cada ponto da fronteira de decisão do perceptron treinado
        no espaço 2D.
        """
        return ((-weight[0]/weight[1])*x+(bias)/weight[1])

# _____________________TREINO_________________________
def treina_modelo():
    """
    Implementa lógica de treinamento em 2D
    -------PARÂMETROS-------
    None.
    ---------RETORNO--------
    w: (float array) vetor de pesos ajustado no treinamento.
    e: (float array) erros quadráticos médios a cada época de treinamento.
    """

    MSEs,stds = cross_validation(base_treino,nfolds=7)

    print(f'ERROS QUADRÁTIOS MÉDIOS - VALIDAÇÃO CRUZADA: {MSEs}')
    print(f'DESVIOS PADRÃO - VALIDAÇÃO CRUZADA: {stds}')

    #plot da função de resultado de treino
    df = pd.read_csv(base_treino)

    #randomiza a base de dados. Pode ser usada para testar a influencia no treino.
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = np.array(df.drop(columns=['CLASSE'])) #variáveis do modelo
    d = np.array(df.drop(columns=['X','Y']))

    w, e = train_model(X, d, nepocas=7,bias=BIAS, learn_rate=0.1,verbose=True)
    print(f'RESULTADO DO TREINAMENTO\nVetor de pesos: {w}\nERROS MÉDIOS QUADRADOS: {e}')

    return w, e

# __________________________TESTE___________________________

def testa_modelo(w):
    """
    Implementa lógica de teste do modelo treinado em 2D.
    ---------PARÂMETROS---------
    w: (float array) vetor de pesos ajustados obtidos do treinamento.
    -----------RETORNO-----------
    X_teste: (array) conjunto de pontos utilizado no teste.
    d_teste: (array) classes de cada objeto presente em X_teste.
    resultados: (array) resultados de classificação de cada elemento de X_teste pela rede perceptron.
    """
    

    df_teste = pd.read_csv(base_teste)


    X_teste = np.array(df_teste.drop(columns=['CLASSE'])) #variáveis do modelo
    d_teste = np.array(df_teste.drop(columns=['X','Y']))

    resultados = perceptron(X_teste, w, 0)
    plot_fronteira_decisao(X_teste,d_teste,w,BIAS,)

    return X_teste, d_teste, resultados

# ____________________AVALIAÇÃO____________________________
def avalia_modelo( d_teste, resultados):
    """
    Obtém o número de classificações corretas do modelo treinado e testado.
    ----------PARÂMETROS------------
    d_teste: (array) classes de cada elemento presente na base de dados de teste.
    resultados: (array) resultados de classificação de cada elemento de teste pela rede perceptron.
    """
    soma = 0
    for i in range(len(d_teste)):
        if resultados[i] == d_teste[i]:
            soma +=1

    print(f'NÚMERO TOTAL DE ELEMENTOS: {len(resultados)}\nCLASSIFICAÇÕES CORRETAS: {soma}')


# _______________________CHECAGEM BASES______________________
def valida_bases():
    """
    Verifica se as bases de dados de treino e teste nao possuem elementos em comum.
    Utilizada para garantir que haja generalização da rede perceptron após
    o treinamento.
    """

    df1 = pd.read_csv(base_treino)
    df2 = pd.read_csv(base_teste)

    df1 = df1.drop(columns=['CLASSE'])
    df2 = df2.drop(columns=['CLASSE'])

    has_common_elements = any(df1.isin(df2).values.flatten())

    if not has_common_elements:
        print("Os DataFrames não possuem elementos em comum.")
    else:
        print("Os DataFrames possuem elementos em comum.")

def plot_fronteira_decisao(X, d, weight, bias,verbose=False):
    """
    Implementa lógica de plotagem da fronteira de decisão de 
    uma rede perceptron treinada em um problema de 2D.
    ---------PARÂMETROS-------------
    X: (array) entradas da rede perceptron.
    d: (array) classes de cada elemento de X.
    weight: (array) vetor de pesos ajustados.
    verbose: (bool, optional) utilizado para salvar, se True,
            ou não salvar a figura plotada, se False.
    """
    n =len(d)

    classe1, classe2 = [], []

    for i in range(n):
        if d[i]==0:
            classe1.append(X[i])
        else:
            classe2.append(X[i])
    classe1 = np.array(classe1)
    classe2 = np.array(classe2)

    x11 = classe1[:,0]
    x21 = classe1[:,1]
    x12 = classe2[:,0]
    x22 = classe2[:,1]
    
    results = np.linspace(x11.min(),x12.max(),100)
    plt.plot(results, fronteira_decisao(results, weight,bias),color='k')
    plt.scatter(x11,x21,color='b', label='CLASSE 0')
    plt.scatter(x12,x22,color='r', label='CLASSE 1')
    plt.xlabel('Variável 1')
    plt.ylabel('Variável 2')
    plt.legend()
    if verbose==True:
        plt.savefig('Resultado_teste.png')
    plt.show()

    
#cria bases de dados
# create_database(2,False)

#verifica se bases de treino e teste não possuem elementos em comum
valida_bases()

#treina modelo e obtem vetor de pesos ajustado e erros médios quadráticos
w, erro_med = treina_modelo()#


#testa o modelo treinado
X, d1, resultados = testa_modelo(w)

#obtem o numero de classificações corretas
avalia_modelo(d1,resultados)

#plota fronteira de decisão 2D
plot_fronteira_decisao(X,d1,w,BIAS,True)

# ----------------------PLOTS RELATORIO-----------------------

# ---------------EVOLUÇÃO DO ERRO DURANTE ÉPOCAS DE TREINAMENTO---------------------------
epocas = np.arange(1,len(erro_med)+1)
plt.plot(epocas,erro_med)
plt.xlabel('Época de treinamento')
plt.ylabel('Erro médio quadrado')
plt.grid()
plt.savefig('MSE.png')
plt.show()

# -------------------------FRONTEIRAS DECISAO TREINAMENTO ---------------------------------
df_balanced = pd.read_csv('src/base_perceptron_balanceada_treino.csv')
df_nls_balanced = pd.read_csv('src/base_perceptron_NLSbalanceada_treino.csv')
df_unbalanced = pd.read_csv('src/base_perceptron_desbalanceada_treino.csv')
df_nls_unbalanced = pd.read_csv('src/base_perceptron_NLSdesbalanceada_treino.csv')

bases = [df_balanced,df_unbalanced,df_nls_balanced,df_nls_unbalanced]
results = []

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(14,12))

i = 0
for base in bases:
    X = np.array(base.drop(columns=['CLASSE'])) #variáveis do modelo
    d = np.array(base.drop(columns=['X','Y']))
    
    classe0 = np.array(base[base['CLASSE']==0].reset_index(drop=True).drop(columns='CLASSE'))
    classe1 = np.array(base[base['CLASSE']==1].reset_index(drop=True).drop(columns='CLASSE'))

    w,e = train_model(X,d,bias=BIAS,nepocas=5, learn_rate=0.1)

    x = np.linspace(classe0[:,0].min(), classe1[:,0].max(),100)
    results.append(fronteira_decisao(x,w,BIAS))

    if i==0:
        ax[0,0].plot(x, results[i],color='k')
        ax[0,0].scatter(classe0[:,0],classe0[:,1], label='classe 0', color='r',marker='x')
        ax[0,0].scatter(classe1[:,0],classe1[:,1], label='classe 1', color='g',marker='x')
        ax[0,0].set_xlabel('Feature 1')
        ax[0,0].set_ylabel('Feature 2')
        ax[0,0].set_title('Treinamento - LS Balanceada')
        ax[0,0].legend()
    elif i==1:
        ax[0,1].plot(x, results[i],color='k')
        ax[0,1].scatter(classe0[:,0],classe0[:,1], label='classe 0', color='r',marker='x')
        ax[0,1].scatter(classe1[:,0],classe1[:,1], label='classe 1', color='g',marker='x')
        ax[0,1].set_xlabel('Feature 1')
        ax[0,1].set_ylabel('Feature 2')
        ax[0,1].set_title('Treinamento - LS Não Balanceada')
        ax[0,1].legend()
    elif i==2:
        ax[1,0].plot(x, results[i],color='k')
        ax[1,0].scatter(classe0[:,0],classe0[:,1], label='classe 0', color='r',marker='x')
        ax[1,0].scatter(classe1[:,0],classe1[:,1], label='classe 1', color='g',marker='x')
        ax[1,0].set_xlabel('Feature 1')
        ax[1,0].set_ylabel('Feature 2')
        ax[1,0].set_title('Treinamento - NLS Balanceada')
        ax[1,0].legend()
    else:
        ax[1,1].plot(x, results[i],color='k')
        ax[1,1].scatter(classe0[:,0],classe0[:,1], label='classe 0', color='r',marker='x')
        ax[1,1].scatter(classe1[:,0],classe1[:,1], label='classe 1', color='g',marker='x')
        ax[1,1].set_xlabel('Feature 1')
        ax[1,1].set_ylabel('Feature 2')
        ax[1,1].set_title('Treinamento - NLS Não Balanceada')
        ax[1,1].legend()
    i+=1
plt.savefig('Fonteiras_treinamento')
plt.show()
# ----------------------------------------------------------------------------------------------

# ------------------------------------MATRIZ DE CONFUSÃO-------------------------------------------
# as bases para treino e teste devem ser alteradas nas variáveis base_treino e base_teste, nas linhas 8 e 9
VP = 0
VN = 0
FP = 0
FN = 0
for classe, resultado in zip(d1,resultados):
    if classe[0]==0 and resultado==0:
        VN +=1
    elif classe[0]==0 and resultado==1:
        FP+=1
    elif classe[0]==1 and resultado==1:
        VP+=1
    else:
        FN+=1

print(FP,VP,FN,VN)
print(f'falsos positivos: {FP}')
print(f'verdadeiros positivos: {VP}')
print(f'falsos negativos: {FN}')
print(f'verdadeiros negativos: {VN}')
print(f'precisão: {VP/(VP+FP)}')
print(f'revocação: {VP/(VP+FN)}')