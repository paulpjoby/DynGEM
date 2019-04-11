import networkx as nx
import keras
from keras.models import Model
from keras.layers  import Dense, Input, Embedding, Reshape,  Lambda
from keras import backend as K, regularizers
import numpy as np
from functools import reduce

dynamic_series = []
activation_fn = 'relu'

#Loss Function for preserving First and Second Order
def build_reconstruction_loss(beta):
    """
    return the loss function for 2nd order proximity
    beta: the definition below Equation 3"""
    assert beta > 1

    def reconstruction_loss(true_y, pred_y):
        diff = K.square(true_y - pred_y)

        # borrowed from https://github.com/suanrong/SDNE/blob/master/model/sdne.py#L93
        weight = true_y * (beta - 1) + 1

        weighted_diff = diff * weight
        return K.mean(K.sum(weighted_diff, axis=1))  # mean square error
    return reconstruction_loss


def edge_wise_loss(true_y, embedding_diff):
    """1st order proximity
    """
    # true_y supposed to be None
    # we don't use it
    return K.mean(K.sum(K.square(embedding_diff), axis=1))  # mean square error


#Loading Snapshot Pickled With networkx 1.11
def loadRealGraphSeries(file_prefix, startId, endId):
    graphs = []
    for file_id in range(startId, endId + 1):
        graph_file = file_prefix + str(file_id) + '_graph.gpickle'
        graphs.append(nx.read_gpickle(graph_file))
    return graphs



#Main Function
embedding_dim = 0  # 1/4 of number of nodes

# Encoding Layer Dims
encoding_dim = []

# Decoding Layer Dims
decoding_dim = [] 

encoding_layers = []
decoding_layers = []


dynamic_series = loadRealGraphSeries('snapshots/s',1,7)

model = None
count = 0

#Initial Values
beta=2
alpha=2
l2_param=1e-3


print("Step 1")
for g in dynamic_series:
    N = g.number_of_nodes()
    count = count + 1
    adj_mat = nx.adjacency_matrix(g).toarray()
    edges = np.array(list(g.edges_iter()))
    weights = [ g[u][v].get('weight',1.0) for u,v in g.edges_iter() ]

    if count == 1 :
        # Create Model from Scratch
        embedding_dim = (int) (N/4) # 1/4 

        input_a = Input(shape=(1,), name = 'input-a-' + str(count), dtype= 'int32')
        input_b = Input(shape=(1,), name = 'input-b-' + str(count), dtype= 'int32')
        edge_weight = Input(shape=(1,), name = 'edge-weight-' + str(count), dtype= 'float32')
        
        # Embedding Layers
        i =  embedding_dim
        while ((i+embedding_dim) < N):
            i = i + embedding_dim
            encoding_dim.append(i)
    
        decoding_dim = encoding_dim[::-1]
        
        print(N)
        print(decoding_dim)
        print(encoding_dim)
        print(embedding_dim)


        embedding_layer = Embedding(output_dim=N, input_dim=N, trainable=False, input_length=1, name='nbr-table')

        embedding_layer.build((None,))    
        embedding_layer.set_weights([adj_mat])   
        encoding_layers.append(embedding_layer)
        encoding_layers.append(Reshape((N,)))  

        encoding_layers_dims = [embedding_dim]

        for i, dim in enumerate(encoding_layers_dims):
            layer = Dense(dim, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='encoding-layer-{}'.format(i))
            encoding_layers.append(layer)

        decoding_layer_dims = encoding_layers_dims[::-1][1:] + [N]
        for i, dim in enumerate(decoding_layer_dims):
            layer = Dense(dim, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='decoding-layer-{}'.format(i))
            decoding_layers.append(layer)

        all_layers =  encoding_layers + decoding_layers

        encoded_a = reduce(lambda arg, f: f(arg), encoding_layers, input_a)
        encoded_b = reduce(lambda arg, f: f(arg), encoding_layers, input_b)

        decoded_a = reduce(lambda arg, f: f(arg), all_layers, input_a)
        decoded_b = reduce(lambda arg, f: f(arg), all_layers, input_b)
        
        subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],
                        output_shape=lambda shapes: shapes[0])

        embedding_diff = subtract_layer([encoded_a,encoded_b])
        embedding_diff = Lambda(lambda x: x * edge_weight)(embedding_diff)

        ####################
        # MODEL
        ####################
        model = Model([input_a, input_b, edge_weight],
                           [decoded_a, decoded_b, embedding_diff])
        
        reconstruction_loss = build_reconstruction_loss(beta)

        model.compile(
	    optimizer='adadelta',
            loss=[reconstruction_loss, reconstruction_loss, edge_wise_loss],
            loss_weights=[1, 1, alpha])
                           



        print("SDNE Model Built")

    else:
        # Create Model from Scratch
        print("Adding More Dynamic Layers")
        input_a = Input(shape=(1,), name = 'input-a-' + str(count), dtype= 'int32')
        input_b = Input(shape=(1,), name = 'input-b-' + str(count), dtype= 'int32')
        edge_weight = Input(shape=(1,), name = 'edge-weight-' + str(count), dtype= 'float32')
        