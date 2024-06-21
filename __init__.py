import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

sep = r'/'

def create_database(n:int=2, balanced:bool=True) -> None:
    """
    Cria base de dados para implementação do perceptron.
    
    --------parâmetros---------
    n: número inteiro de classes desejadas, padronizado em 2.
    balanced: variável booleana. True se as classes devem ser balanceadas,
              False caso contrário.
    """
    x_mean = np.random.randint(-10,10)
    y_mean = np.random.randint(-10,10)

    rad_x = np.random.randint(1,2)
    rad_y = np.random.randint(1,2)

    df = pd.DataFrame()
    for i in range(n):
        aux = pd.DataFrame(columns=['X', 'Y', 'CLASSE'])
        if i!=0:
            soma = np.random.randint(8,10)

            x_mean += 4.5*soma
            y_mean += 4.6*soma

        mean = [x_mean, y_mean]
        cov = [[rad_x, 0], [0, rad_y]]

        num = 70

        if balanced==False:
            if i%2!=0:
                num = np.random.randint(2,6)
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


    df.to_csv(os.getcwd()+sep+'src'+sep+'base_perceptron_'+ name+'.csv', index=False)

    plt.axis('equal')
    plt.grid()
    plt.show()

    return 

def activation_func(v):
    return 1 if v>=0 else 0

def train_model(X:np.array, Y:np.array, bias:float=0, learn_rate:float=0.01):
    """
    Realiza o treinamento do perceptron.
    -------PARÂMETROS------
    X: (array dimensão n x 2) contém os valores de variáveis usadas na classificação
    Y: (array dimensão n) contém as classes de cada objeto.
    learn_rate: (float) taxa de aprendizado do perceptron. Valor padrão 0.01
    bias: (float) bias do perceptron. Valor padrão 0
    """
    lin,col = np.shape(X)
    w = np.zeros(col)

    for entrada, objetivo in zip(X, Y):
        y = activation_func(np.matmul(w,entrada.T) + bias)
        w = w + learn_rate*(objetivo - y)* entrada
    
    return w, bias

def cross_validation(base:str):
    """
    Implementa validação cruzada para bases de dados estratificadas,
    utilizando o StratifiedKFold para a construção da base.
    Numero de dobras padronizado em 5.
    ------PARÂMETROS------
    base: (string) caminho para a base primária para gerar as bases da validação cruzada.
    """

    from sklearn.model_selection import StratifiedKFold

    df = pd.read_csv(base, sep=',', decimal='.')

    X = np.array(df.drop(columns=['CLASSE'])) #variáveis do modelo
    Y = np.array(df.drop(columns=['X','Y'])) #classes do modelo

    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(base)

    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        # print(f"Fold {i}:")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")

        train_model(X, Y)
        break

# create_database(2, False)

base = 'src/base_perceptron_balanceada_treino.csv'
cross_validation(base)


