import csv
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import odeint
import seaborn as sns
from sklearn.linear_model import LinearRegression

caminho_arquivo = '/content/VRA_2022_01.csv'

df = pd.read_csv(caminho_arquivo, sep= "\t")

df.head()

df.rename(columns={'Descrição Aeroporto Origem': 'origem'}, inplace=True)
df.rename(columns={'Descrição Aeroporto Destino': 'destino'}, inplace=True)
df.rename(columns={'Situação Voo': 'situacao'}, inplace=True)
df.head()

df = df[df['situacao'] != 'CANCELADO']
df.head()

total_assentos = df['Número de Assentos'].sum()



> O código percorre um DataFrame para contar rotas entre aeroportos, cria uma matriz de adjacência para representar essas rotas e utiliza a biblioteca NetworkX para construir e visualizar um grafo, onde os nós representam aeroportos e as arestas representam as rotas entre eles, ponderadas pelo número de ocorrências de cada rota.





rotas = defaultdict(int)

for index, row in df.iterrows():
    origem = row['origem']
    destino = row['destino']
    rotas[(origem, destino)] += 1

aeroportos = sorted(set().union(*[set(rota) for rota in rotas.keys()]))

matriz_adjacencia = np.zeros((len(aeroportos), len(aeroportos)), dtype=int)
for i, origem in enumerate(aeroportos):
    for j, destino in enumerate(aeroportos):
        matriz_adjacencia[i, j] = rotas.get((origem, destino), 0)

print("Matriz de Adjacência:")
df_matriz = pd.DataFrame(matriz_adjacencia, index=aeroportos, columns=aeroportos)
print(df_matriz)

G = nx.Graph()

for aeroporto in aeroportos:
    G.add_node(aeroporto)

for (origem, destino), contagem in rotas.items():
    G.add_edge(origem, destino, weight=contagem)

pos = nx.random_layout(G)

plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=False, node_size=500, node_color='skyblue', font_size=10, edge_color='gray', width=1.5, arrowsize=20)
plt.title('Grafo de Rotas entre Aeroportos')
plt.show()



> Criação e visualização de um subgrafo dos cinco primeiros aeroportos do grafo original, destacando as rotas entre eles. Cada aresta no subgrafo possui um rótulo que indica o número de voos (ou a contagem de rotas) entre os aeroportos conectados. A idéia é facilitar a compreensão das conexões e a densidade das rotas entre os primeiros cinco aeroportos.




GP = nx.Graph()
for (origem, destino), contagem in rotas.items():
    GP.add_edge(origem, destino, weight=contagem, label=contagem)

cinco_nos = list(GP.nodes())[:5]

subgrafo = GP.subgraph(cinco_nos)

pos = nx.random_layout(subgrafo)

plt.figure(figsize=(10, 6))

nx.draw_networkx_edges(subgrafo, pos, width=1.0, alpha=0.5)

nx.draw_networkx_nodes(subgrafo, pos, node_size=200, node_color='skyblue')

edge_labels = nx.get_edge_attributes(subgrafo, 'label')
nx.draw_networkx_edge_labels(subgrafo, pos, edge_labels=edge_labels, font_size=8)

nx.draw_networkx_labels(subgrafo, pos, font_size=10, font_color='black')

plt.title('Grafo de Rotas entre Aeroportos com Valores e os primeiros 5 nós')
plt.axis('off')
plt.show()




> Calculo da centralidade de grau




degree = nx.degree_centrality(G)

df_degree = pd.DataFrame(degree.items(), columns=['Nó', 'Centralidade de Grau'])

df_degree_sorted = df_degree.sort_values(by='Centralidade de Grau', ascending=False)

print("DataFrame da Centralidade de Grau (Ordenado):")
print(df_degree_sorted)




> boxplot dos valores de centralidade de grau




plt.figure(figsize=(8, 6))
sns.boxplot(y='Centralidade de Grau', data=df_degree_sorted)
plt.title('Boxplot da Centralidade de Grau')
plt.ylabel('Centralidade de Grau')
plt.show()

print("\nEstatísticas Descritivas:")
print(stats.describe(list(degree.values())))

# **Modelo SIR**

G = nx.from_pandas_edgelist(df, 'origem', 'destino', create_using=nx.DiGraph())

for u, v, d in G.edges(data=True):
    d['weight'] = len(df[(df['origem'] == u) & (df['destino'] == v)])

aeroportos_validos = set(G.nodes())
aeroportos_com_assentos = set(df['origem'])
aeroportos_com_dados = aeroportos_validos.intersection(aeroportos_com_assentos)

N = {aeroporto: total_assentos for aeroporto in aeroportos_com_dados}

N_realista = total_assentos // 100

I0 = {aeroporto: N_realista * 0.01 for aeroporto in aeroportos_com_dados}
S0 = {aeroporto: N[aeroporto] - I0[aeroporto] for aeroporto in aeroportos_com_dados}
R0 = {aeroporto: 0 for aeroporto in aeroportos_com_dados}

beta = 0.3 
gamma = 0.5

def modelo_sir(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

dias_janeiro = range(1, 32)
t = np.linspace(0, len(dias_janeiro) - 1, len(dias_janeiro))

resultados = {}
for aeroporto in aeroportos_com_dados:
    y0 = S0[aeroporto], I0[aeroporto], R0[aeroporto]
    resultados[aeroporto] = odeint(modelo_sir, y0, t, args=(N_realista, beta, gamma))

plt.figure(figsize=(10, 6))
for aeroporto, resultado in resultados.items():
    S, I, R = resultado.T
    plt.plot(dias_janeiro, I, label=aeroporto)

plt.xlabel('Dias de Janeiro')
plt.ylabel('Número de Infectados')
plt.title('Simulação do Modelo SIR em Aeroportos - Janeiro de 2022')
plt.gca().ticklabel_format(style='plain', axis='y')
plt.grid(True)
plt.show()

# **Regressão Linear**

no_mais_central = df_degree_sorted.iloc[0]['Nó']

N = df[df['origem'] == no_mais_central]['Número de Assentos'].sum()

I0 = N * 0.01 
S0 = N - I0
R0 = 0

dias_fevereiro = np.arange(1, 29)

t_fevereiro = np.linspace(0, len(dias_fevereiro) - 1, len(dias_fevereiro))

y0 = S0, I0, R0
resultado_janeiro = odeint(modelo_sir, y0, t, args=(N, beta, gamma))
S_janeiro, I_janeiro, R_janeiro = resultado_janeiro.T

dias_totais = np.concatenate((dias_janeiro, dias_fevereiro)).reshape(-1, 1)
I_totais = np.concatenate((I_janeiro, np.zeros(len(dias_fevereiro)))).reshape(-1, 1)

modelo = LinearRegression()
modelo.fit(dias_totais, I_totais)
predicao_fevereiro = modelo.predict(dias_fevereiro.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(dias_janeiro, I_janeiro, label=f'{no_mais_central} - Real (Janeiro)', color='blue')
plt.plot(dias_fevereiro, predicao_fevereiro, label=f'{no_mais_central} - Predição (Fevereiro)', color='red', linestyle='--')

plt.xlabel('Dias de Fevereiro')
plt.ylabel('Número de Infectados')
plt.title('Previsão do Número de Infectados para Fevereiro de 2022')
plt.xticks(np.arange(1, 29, step=1))
plt.legend()
plt.grid(True)
plt.show()
