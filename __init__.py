import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

sep = r'/'

base_treino = 'src/base_perceptron_balanceada_treino.csv'
base_teste = 'src/base_perceptron_balanceada_teste.csv'

def create_database(n:int=2, balanced:bool=True) -> None:
    """
    Cria base de dados para implementação do perceptron.
    
    --------parâmetros---------
    n: número inteiro de classes desejadas, padronizado em 2.
    balanced: variável booleana. True se as classes devem ser balanceadas,
              False caso contrário.
    """
    x_mean = -1
    y_mean = -1

    rad_x = 2.3
    rad_y = 2.3

    df = pd.DataFrame()
    for i in range(n):
        aux = pd.DataFrame(columns=['X', 'Y', 'CLASSE'])
        if i!=0:
            soma = 8

            x_mean += 1.55*soma
            y_mean += 1.55*soma

        mean = [x_mean, y_mean]
        cov = [[rad_x, 0], [0, rad_y]]

        num = 20

        if balanced==False:
            if i%2!=0:
                num = 9
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


    df.to_csv(os.getcwd()+sep+'src'+sep+'base_perceptron_'+ name+'_teste.csv', index=False)

    plt.axis('equal')
    plt.grid()
    plt.show()

    return 

def activation_func(v):
    return 1 if v>=1.5 else 0

def perceptron(X, d, w, bias):
    result = []
    for entrada, objetivo in zip(X, d):
        y = activation_func(np.matmul(w,entrada)+bias)
        result.append(y)
    result = np.array(result)
    return result


def cross_validation(base:str, nfolds=10):
    """
    Implementa validação cruzada para bases de dados estratificadas,
    utilizando o StratifiedKFold para a construção da base.
    Numero de dobras padronizado em 10.
    ------PARÂMETROS------
    base: (string) caminho para a base primária para gerar as bases da validação cruzada.
    """

    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils import shuffle

    df = pd.read_csv(base, sep=',', decimal='.')

    X = np.array(df.drop(columns=['CLASSE'])) #variáveis do modelo
    Y = np.array(df.drop(columns=['X','Y'])) #classes do modelo

    X, Y = shuffle(X,Y,random_state=42)

    skf = StratifiedKFold(n_splits=nfolds)
    skf.get_n_splits(base)

    MSEs = []

    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        bias = 0
        X_treino = np.array(df.drop(columns=['CLASSE']).iloc[train_index])
        Y_treino = np.array(df.drop(columns=['X','Y']).iloc[train_index])
        w = train_model(X_treino, Y_treino, bias=bias,learn_rate=0.01, nepocas=100)
        
        pesos_teste = w
        
        X_teste = np.array(df.drop(columns=['CLASSE']).iloc[test_index])
        Y_teste = np.array(df.drop(columns=['X','Y']).iloc[test_index])

        resultados = perceptron(X_teste,Y_teste, pesos_teste, bias)

        # calculo do erro quadratico medio
        erro_quad = 0
        for i in range(len(resultados)):
            erro_quad = erro_quad+ (Y_teste[i] - resultados[i])**2

        MSE = erro_quad/len(resultados)
        MSEs.append(MSE)

    MSEs = np.array(MSEs)
    return MSEs

def train_model(X:np.array, Y:np.array, bias:float=0, learn_rate:float=0.01,nepocas:int=10):
    """
    Realiza o treinamento do perceptron.
    -------PARÂMETROS------
    X: (array dimensão n x 2) contém os valores de variáveis usadas na classificação
    Y: (array dimensão n) contém as classes de cada objeto.
    learn_rate: (float) taxa de aprendizado do perceptron. Valor padrão 0.01
    bias: (float) bias do perceptron. Valor padrão 0
    """
    lin,col = np.shape(X)
    w = np.full(col, 0)

    for i in range(nepocas):    
        for entrada, objetivo in zip(X, Y):
            y = activation_func(np.matmul(w,entrada)+bias)
            w = w + learn_rate*(objetivo - y)* entrada

    return w
# _____________________TREINO_________________________
def treina_modelo():

    MSEs = cross_validation(base_treino,5)

    print(F'ERROS QUADRÁTIOS MÉDIOS - VALIDAÇÃO CRUZADA: {MSEs}')

    #plot da função de resultado de treino
    df = pd.read_csv(base_treino).sort_index()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = np.array(df.drop(columns=['CLASSE'])) #variáveis do modelo
    d = np.array(df.drop(columns=['X','Y']))

    w = train_model(X, d, nepocas=100,bias=0, learn_rate=0.01)
    print(f'RESULTADO DO TREINAMENTO: {w}')

    return w

# __________________________TESTE___________________________

def testa_modelo(w):
    

    df_teste = pd.read_csv(base_teste)


    X_teste = np.array(df_teste.drop(columns=['CLASSE'])) #variáveis do modelo
    d_teste = np.array(df_teste.drop(columns=['X','Y']))

    resultados = perceptron(X_teste, d_teste, w, 0)

    return X_teste, d_teste, resultados

# ____________________AVALIAÇÃO____________________________
def avalia_modelo(X_teste, d_teste, resultados):
    soma = 0
    for i in range(len(d_teste)):
        if resultados[i] == d_teste[i]:
            soma +=1

    print(f'NÚMERO TOTAL DE ELEMENTOS: {len(resultados)}\nCLASSIFICAÇÕES CORRETAS: {soma}')


# _______________________CHECAGEM BASES______________________
def valida_bases():
    df1 = pd.read_csv(base_treino)
    df2 = pd.read_csv(base_teste)

    df1 = df1.drop(columns=['CLASSE'])
    df2 = df2.drop(columns=['CLASSE'])

    has_common_elements = any(df1.isin(df2).values.flatten())

    if not has_common_elements:
        print("Os DataFrames não possuem elementos em comum.")
    else:
        print("Os DataFrames possuem elementos em comum.")

# create_database(2)

valida_bases()


w = treina_modelo()#vetor de pesos obtido do treino do modelo

X, d, resultados = testa_modelo(w)

avalia_modelo(X,d,resultados)

