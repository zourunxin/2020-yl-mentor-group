import networkx as nx               #载入networkx包
import matplotlib.pyplot as plt     #用于画图
G = nx.Graph()                     #无向图
example_edges = [('A','B'),('C','E'),('D','E'), 
                 ('F','G'),('F','H'),('G','I'), 
                 ('G','J'),('H','J'),('H','L'), 
                 ('H','M'),('I','K'),('J','K'), 
                 ('L','K'),('L','H'),('L','M')]
G.add_edges_from(example_edges)
print(nx.is_connected(G))
print(nx.number_connected_components(G))
for i in nx.connected_components(G):
    print(i)
graphs=list(nx.connected_component_subgraphs(G))
plt.plot(graphs)