import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

sep = r'/'

def create_database(n=2, balanced=True):
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

    print(os.getcwd() + sep + 'base_perceptron_' + name + '.csv')
    df.to_csv(os.getcwd() + sep + 'base_perceptron_' + name + '.csv', index=False)

    plt.axis('equal')
    plt.grid()
    plt.show()

    return 

create_database(2, False)