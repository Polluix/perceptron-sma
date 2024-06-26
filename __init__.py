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
    Cria base de dados para implementação do perceptron.
    
    --------parâmetros---------
    n: número inteiro de classes desejadas, padronizado em 2.
    balanced: variável booleana. True se as classes devem ser balanceadas,
              False caso contrário.
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

def activation_func(v):
    return 1 if v>=BIAS else 0

def perceptron(X, w, bias):
    result = []
    for entrada in X:
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

    MSE = np.ones(nfolds)

    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
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
        

    return MSE

def train_model(X:np.array, Y:np.array, bias:float=0, learn_rate:float=0.01,nepocas:int=10,verbose=False):
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

def fronteira_decisao(x, weight, bias):
        return ((-weight[0]/weight[1])*x+(bias)/weight[1])

# _____________________TREINO_________________________
def treina_modelo():

    MSEs = cross_validation(base_treino,nfolds=7)

    print(F'ERROS QUADRÁTIOS MÉDIOS - VALIDAÇÃO CRUZADA: {MSEs}')

    #plot da função de resultado de treino
    df = pd.read_csv(base_treino)
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = np.array(df.drop(columns=['CLASSE'])) #variáveis do modelo
    d = np.array(df.drop(columns=['X','Y']))

    w, e = train_model(X, d, nepocas=7,bias=BIAS, learn_rate=0.1,verbose=True)
    print(f'RESULTADO DO TREINAMENTO\nVetor de pesos: {w}\nERROS MÉDIOS QUADRADOS: {e}')

    return w, e

# __________________________TESTE___________________________

def testa_modelo(w):
    

    df_teste = pd.read_csv(base_teste)


    X_teste = np.array(df_teste.drop(columns=['CLASSE'])) #variáveis do modelo
    d_teste = np.array(df_teste.drop(columns=['X','Y']))

    resultados = perceptron(X_teste, w, 0)
    plot_fronteira_decisao(X_teste,d_teste,w,BIAS,)

    return X_teste, d_teste, resultados

# ____________________AVALIAÇÃO____________________________
def avalia_modelo( d_teste, resultados):
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

def plot_fronteira_decisao(X, d, weight, bias,verbose=False):
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

    

# create_database(2,False)

valida_bases()


w, erro_med = treina_modelo()#vetor de pesos obtido do treino do modelo



X, d1, resultados = testa_modelo(w)

avalia_modelo(d1,resultados)

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

# ------------------------------------matriz de confusão-------------------------------------------
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