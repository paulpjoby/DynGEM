#Dynamic Network Embedding
- Btech S8 Project (2019)
###System Specifications
- Python 3.6
- Operating Systems Tested : Windows 10 or Ubuntu 18.04 LTS

###Packages
- Babel 2.6.0
- backcall 0.1.0
- conda 4.6.8
- conda-build 3.17.6
- Cython 0.29.2
- h5py 2.8.0
- Keras <= 2.2.4
- Keras-Applications 1.0.7
- Keras-Preprocessing 1.0.9
- networkx 1.11 (Only this Version)
- scikit-learn 0.20.1
- scipy 1.1.0
- seaborn 0.9.0
- six 1.12.0
- tensorboard 1.13.1
- tensorflow 1.13.1
- tensorflow-estimator 1.13.0
- Theano 1.0.3+2.g3e47d39ac.dirty
- tqdm 4.28.1

### Running Dynamic Embedding Network Embedding

- Pickling Dataset Snapshots (To make it ready to load for embedding)
`$ python datasets/<name_of_dataset>/graph_pickler.py`

- Performing Dynamic Embedding for a Specific Dataset
`$ python datasets/<name_of_dataset>/dyngem.py`

- Performing Link Prediction on Embedded Graph
`$ python datasets/<name_of_dataset>/link_prediction.py`


### Dataset Plots
![](https://github.com/paulpjoby/DynGEM/blob/master/Results/datasets.png)

### Dynamic Graph Embedding Architecture 
![](https://github.com/paulpjoby/DynGEM/blob/master/Results/Proposed%20Model.png)

### Performance Comparisons with SDNE
![](https://github.com/paulpjoby/DynGEM/blob/master/Results/Performance%20Comparison%20Graph%20Main.png)

### Parameter Sensitivity
![](https://github.com/paulpjoby/DynGEM/blob/master/Results/Number%20of%20encoding%20layers%20vs%20AUC.png)

![](https://github.com/paulpjoby/DynGEM/blob/master/Results/Number%20of%20snapshots%20vs%20%20AUC.png)

![](https://github.com/paulpjoby/DynGEM/blob/master/Results/Embedding%20Dimensions%20vs%20AUC.png)

