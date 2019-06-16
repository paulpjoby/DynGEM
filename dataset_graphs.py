'''NOTE : This file is used for plotting the graphs of the dataset
'''
import matplotlib.pyplot as plt
import networkx as nx

#Read the graph to draw
# g = nx.read_('datasets/haggle_snapshots/snapshots/s7_graph.gpickle')
G = nx.read_edgelist('datasets/hep-th_snapshot/snapshots/s8.txt', create_using=nx.Graph())
nx.draw_networkx(G, with_labels = False, node_size=10, node_color='r')

plt.show()
