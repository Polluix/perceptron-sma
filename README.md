
# perceptron-sma
Implementação de perceptron para a disciplina de Sistemas Inteligentes.

Todas funções foram documentadas, contendo os detalhes de cada variável e
seu comportamento no código.

Toda a lógica é implementada no arquivo __init__.py.

#---------------------INSTALAÇÃO DAS DEPENDENCIAS NECESSÁRIAS-------------------

O projeto foi implementado utilizando um ambiente virtual e as dependências foram instaladas
utilizando o pip. Para a criação de um ambiente virtual utilize o comando

                        python -m venv .venv

e para a ativação do mesmo execute o script de ativação com o comando

                        .venv\Scripts\activate

em seguida, instale as dependências necessárias utilzando o comando

                     pip install -r requirements.txt

#-------------------INSTRUÇÕES DE EXECUÇÃO----------------------------

1. Utilize a função create_base() para criar bases de dados de trieno e teste para o perceptron.
As bases utilizadas estão presentes na pasta /src. Caso novas bases sejam criadas, é importante
alterar o nomes destas, como descrito na documentação da função dentro do código.

2. A função valida_bases() é necessária para verificar se as bases de treino e teste não possuem
   uma interseção, garantindo a generalização do perceptron.

3. As funções de treino e teste da rede são treina_modelo() e testa_modelo(), respectivamente, já
   implementadas no arquivo __init__.py

4. As funções de plotagem só funcionam para problemas de classificação de duas dimensões, dada a
   facilidade de visualizar os dados e a fronteira de decisão dentro do espaço nessas dimensões.
   Tais funções devem ser comentadas caso ocorra a necessidade de treinar o perceptron para um
   problema de classificação de mais de duas dimensões.

6. Após a definição de todas as constantes e funções é feita a implementação da lógica de treino,
   teste e avaliação do perceptron, assim como os plots utilizados no relatório.

7. Detalhes sobre alterações na criação de bases de dados e utilização dos parâmetros da rede
   são explicados nos comentários das funções implementadas.

