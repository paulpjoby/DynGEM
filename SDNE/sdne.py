import networkx as nx
import keras
from keras.models import Model, Sequential, load_model
from keras.layers  import Dense, Input, Embedding, Reshape,  Lambda
from keras import backend as K, regularizers
import numpy as np
from functools import reduce
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dynamic_series = []
activation_fn = 'sigmoid'
activation_fn_embedding_layer='sigmoid'
loss_function = 'binary_crossentropy'


#Loading Snapshot Pickled With networkx 1.11
def loadRealGraphSeries(file_prefix, startId, endId):
    graphs = []
    for file_id in range(startId, endId + 1):
        graph_file = file_prefix + str(file_id) + '_graph.gpickle'
        graphs.append(nx.read_gpickle(graph_file))
    return graphs

def get_encoder(model, input_name):
    #Encoder Model
    model = model.model
    encoder = Sequential()
    flag = 0 # Keep Adding layer to Sequential 
    for layer in model.layers:
        if flag == 1:
            break
        if layer.name == 'embedding-layer':
            flag = 1
        encoder.add(layer)
        
    encoder = Model(encoder.input, encoder.get_layer('embedding-layer').output)
    return encoder

def get_decoder(model):
    decoder = None
    return decoder

def link_prediction():
    file_prefix = 'graphs/mit/s'
    graph_file = file_prefix + str(8) + '_graph.gpickle'
    g = nx.read_gpickle(graph_file)
    np.set_printoptions(threshold=np.nan)
    N = g.number_of_nodes()

    test_ratio = 0.15
    train_set, test_edges = train_test_split(g.edges(), test_size=test_ratio)
    #g.remove_edges_from(test_edges)

    final_model = load_model("models/final.h5")

    adj_mat = nx.adjacency_matrix(g).toarray()

    reconstructed_adj = final_model.predict(adj_mat)

    # reconstructed_adj = np.reshape(reconstructed_adj, (-1, 2))
    # adj_mat = np.reshape(adj_mat, (-1, 2))

    rows = adj_mat.shape[0]
    cols = adj_mat.shape[1]
    y_act = []

    for x in range(0, rows):
        for y in range(0, cols):
            y_act.append(adj_mat[x,y])

    pred =  []   
    for x in range(0, rows):
        for y in range(0, cols):
            pred.append(reconstructed_adj[x,y])


    fpr, tpr, thresholds = metrics.roc_curve(y_act, pred)
    print(metrics.auc(fpr, tpr))
    roc_auc = metrics.auc(fpr, tpr)


    #Plotting

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return None

def get_embedding(encoder, graph_name, output_name):
    g = nx.read_edgelist(graph_name, create_using=nx.Graph())
    g = nx.convert_node_labels_to_integers(g)
    graph = g
    N = graph.number_of_nodes()
    adj_mat = nx.adjacency_matrix(graph).toarray()
    embedding = encoder.predict(adj_mat)
    np.savetxt(output_name,embedding)
    return embedding

def build_model():
    #Main Function
    embedding_dim = 0  # 1/4 of number of nodes
    # Encoding Layer Dims
    encoding_dim = []
    # Decoding Layer Dims
    decoding_dim = [] 
    encoding_layers = []
    decoding_layers = []
    dynamic_series = loadRealGraphSeries('graphs/mit/s',8,8)
    count = 0
    node_count = []
    #Initial Values
    beta=2
    alpha=2
    l2_param=1e-3
    print("Builiding Model.................")
    final_model = None
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
            
            print("Number Of Nodes: ", end=" ")
            print(N)
            print("Decoding Dimensions : ", end=" ")
            print(decoding_dim)
            print("Encoding Dimensions : ", end=" ")
            print(encoding_dim)
            print("Embedding Dimension : ", end=" ")
            print(embedding_dim)

            #Initializing Model
            model = Sequential()

            i = 0
            for dim in encoding_dim:
                i = i + 1
                layer = Dense(dim, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='encoding-layer-{}'.format(i))
                model.add(layer)

            
            model.add(Dense(embedding_dim, activation=activation_fn_embedding_layer, kernel_regularizer=regularizers.l2(l2_param), name='embedding-layer'))
            
            i = 0
            decoding_dim.append(N)
            for dim in decoding_dim:
                i = i + 1
                layer = Dense(dim, activation=activation_fn, kernel_regularizer=regularizers.l2(l2_param), name='decoding-layer-{}'.format(i))
                model.add(layer)



            model.compile(loss=loss_function, optimizer='adam', metrics=['acc','mae'])
            model.fit(adj_mat,adj_mat,epochs=100)
            model_name = 'models/final.h5'

            model.summary()
            model.save(model_name)

            print("SDNE Initial Model Built Completed ... ")
            final_model = model # Saving The Final Model

        
    return final_model,'dynamic-encoding-layer-{}'.format(count) #Returning the final model
            
        
f_model, input_layer_name = build_model()
f_model.summary()

link_prediction()
print("******************* Embedding ************************")

# encoder = get_encoder(f_model, input_layer_name)
# get_embedding(encoder,"graphs/final_graph.txt", "embedding/final_output.embedding")

