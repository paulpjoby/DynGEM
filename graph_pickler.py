import numpy as np
import networkx as nx
for i in range(1,8):
	filename = "s" + str(i) + ".txt"
	nfilename = "s" + str(i) + "_graph.gpickle"
	G = nx.read_edgelist(filename, create_using=nx.Graph())
	nx.write_gpickle(G,nfilename)
