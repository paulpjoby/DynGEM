import networkx as nx
import keras
from keras.models import Model,Sequential, load_model
from keras.layers  import Dense, Input, Embedding, Reshape,  Lambda
from keras import backend as K, regularizers
import numpy as np
from functools import reduce



model =  Sequential()
model.add(Dense(4, input_dim = 5))
model.summary()
model.save('prev_model.h5')
model = model.model
modelx = Sequential()
for l in model.layers:
    modelx.add(l)

modelx.summary()