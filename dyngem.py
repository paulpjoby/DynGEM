import networkx as nx
import keras
from keras.models import Model, Sequential, load_model
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

def get_encoder():
    return None

#Main Function
embedding_dim = 0  # 1/4 of number of nodes

# Encoding Layer Dims
encoding_dim = []

# Decoding Layer Dims
decoding_dim = [] 

encoding_layers = []
decoding_layers = []


dynamic_series = loadRealGraphSeries('snapshots/s',1,7)


count = 0
node_count = []
#Initial Values
beta=2
alpha=2
l2_param=1e-3


print("Step 1")
for g in dynamic_series:
    N = g.number_of_nodes()
    node_count.append(N)
    count = count + 1
    adj_mat = nx.adjacency_matrix(g).toarray()
    edges = np.array(list(g.edges_iter()))
    weights = [ g[u][v].get('weight',1.0) for u,v in g.edges_iter() ]

    if count == 1 :
        # Create Model from Scratch
        embedding_dim = (int) (N/4) # 1/4 

        # Embedding Layers
        i =  embedding_dim
      
        while ((i+embedding_dim) < N):
            i = i + embedding_dim
            encoding_dim.append(i)
        
        encoding_dim.append(N)

        decoding_dim = encoding_dim
        encoding_dim = encoding_dim[::-1]
        
        print(N)
        print(decoding_dim)
        print(encoding_dim)
        print(embedding_dim)

        #Initializing Model
        model = Sequential()

        i = 0
        for dim in encoding_dim:
            i = i + 1
            layer = Dense(dim, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='encoding-layer-{}'.format(i))
            model.add(layer)

        model.add(Dense(embedding_dim, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='embedding-layer'))

        i = 0
        decoding_dim.append(N)
        for dim in decoding_dim:
            i = i + 1
            layer = Dense(dim, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='decoding-layer-{}'.format(i))
            model.add(layer)



        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        model.fit(adj_mat,adj_mat,epochs=10)
        model_name = 'models/prev_model_{}.h5'.format(count)

        model.summary()
        model.save(model_name)

        print("SDNE Model Built")

    else:
        # Create Model from Scratch
        print("Adding More Dynamic Layers")
        prev_N = node_count[count - 2]
        N = g.number_of_nodes()
        prev_model_name = 'models/prev_model_{}.h5'.format(count-1)
        curr_model_name = 'models/prev_model_{}.h5'.format(count)

        if prev_N == N:  
            prev_model = load_model(prev_model_name)
            prev_model.save(curr_model_name)
            continue

        if prev_N < N:
            prev_model = load_model(prev_model_name)
        
            input_layer = Dense(N,input_dim=N,activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='dynamic-encoding-layer-{}'.format(count))
            input_layer_dummy = Dense(prev_N, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='dynamic-encoding-layer-support-{}'.format(count))
            output_layer = Dense(N, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='dynamic-decoding-layer-{}'.format(count))
            
            curr_model = Sequential()

            #Adding the New Input Layer
            curr_model.add(input_layer)
            curr_model.add(input_layer_dummy)

            #Adding the Existing Layers
            model_api = prev_model.model
            for layer in model_api.layers:
                curr_model.add(layer)
        
            #Adding the Existing Layer
            curr_model.add(output_layer)

            curr_model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
            curr_model.fit(adj_mat,adj_mat,epochs=10)

            #Sequential Before Saving
            curr_model.summary()
            curr_model.save(curr_model_name)
            print("Dynamic {}".format(count-1))
            


