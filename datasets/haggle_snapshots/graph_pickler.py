import numpy as np
import networkx as nx
for i in range(1,8):
	filename = 'snapshots/s' + str(i) + ".txt"
	nfilename = 'snapshots/s' + str(i) + "_graph.gpickle"
	G = nx.read_edgelist(filename, create_using=nx.Graph())
	print(G.number_of_nodes())
	nx.write_gpickle(G,nfilename)
