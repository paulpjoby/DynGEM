'''NOTE : This file is used for plotting the graphs of the dataset
'''
import matplotlib.pyplot as plt
import networkx as nx

#Read the graph to draw
g = nx.read_gpickle('snapshots/s7_graph.gpickle')

nx.draw_networkx(g)
plt.show()
