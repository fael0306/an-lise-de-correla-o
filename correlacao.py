import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as lr

# Área~Preço (m2~price)

casas = pd.read_csv("C:\\Users\\f0fp1107\\Documents\\casas.csv")

# Zerando os valores nulos
casas = casas.fillna(0)

# Escolhendo casas aleatórias por amostragem aleatória simples de 1% dos dados
casas = casas.sample(n=int(round(len(casas)*0.01)),replace=False,random_state=1)

# Verificando mapa de correlações
# 0.9 para mais ou para menos indica uma correlação muito forte.
# 0.7 a 0.9 positivo ou negativo indica uma correlação forte.
# 0.5 a 0.7 positivo ou negativo indica uma correlação moderada.
# 0.3 a 0.5 positivo ou negativo indica uma correlação fraca.
# 0 a 0.3 positivo ou negativo indica uma correlação desprezível.
sns.heatmap(casas.corr(), annot=True)
plt.show()

# Achando coeficientes da regressão linear
a1,a2=lr.linear_regression(casas['m2'],casas['price'])

# Buscando x e y dos gráficos
x = casas['m2']
y = casas['price']

# Plotando os gráficos
g = sns.scatterplot(x=x,y=y)
g = sns.lineplot(x=x,y=a1*x+a2)
plt.show()