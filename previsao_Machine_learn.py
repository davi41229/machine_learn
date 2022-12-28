# Previsão de Receitas com Machine Learn usando modelo de regressão linear com a biblioteca sklear

import matplotlib.pyplot as plt #biblioteca para criação de gráfico

import numpy as np # biblioteca para calculos matematicos

import sklearn # biblioteca para machine learn

# atribuindo as listas nas variaveis
meses = [[1],[2],[3],[4],[5],[6]]
receitas = [[5000],[7000],[7500],[8000],[8700],[9200]]

# contruindo um grafico dos dados com matplotlib

plt.figure() # CHAMAR UMA FIGURA COM MATPLOTLIB
plt.xlabel('meses do ano (Ordem dos meses)') # EIXO X
plt.ylabel('receita obtida (receita)') # EIXO Y
plt.title('mês x receita') # TITULO DO GRAFICO
plt.plot(meses, receitas, 'k') # SINAL QUE MOSTRA OS PONTOS
plt.axis([1,12,5000,15000]) # LIMITES DOS EIXOS
plt.grid(True) # GERAR UM GRID COM LINHAS NO GRAFICO
plt.show() # MOSTRAR O GRAFICO


# importando modelo de regressão linear do sklearn
from sklearn.linear_model import LinearRegression

# Armazenando os dados das listas nas variaveis
X = [[1],[2],[3],[4],[5],[6]]
Y = [[5000],[7000],[7500],[8000],[8700],[9200]]

# Criando o modelo(instanciando a classe linearRegression do sklearn)
modelo = LinearRegression()

# Treinando o modelo com a função fit()
modelo.fit(X,Y)

# Chamar a função (predict) para fazer a previsão
print("PREVISÃO: No setimo mês, a receita obtida será: %.2f"%modelo.predict([[7]]))

# fIM DO ALGORITMO 
print( 10 *"*" + "FIM DA PREVISÃO USANDO linguagem python e machine-learn com regressão-linear e sklearn." + 10 *"*")
