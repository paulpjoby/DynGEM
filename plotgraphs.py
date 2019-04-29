import matplotlib.pyplot as plt
import pandas as pd 

#DynGEM Vs SDNE
index = ['Haggle', 'Hep-th', 'MIT']
sdne = [0.9, 0.58, 0.81]	
dyngem = [0.83 , 0.7, 0.86 ]

df = pd.DataFrame({'SDNE':sdne, 'DynGEM':dyngem}, index = index)
ax = df.plot.bar(rot = 0, width = 0.75)
plt.xlabel('Datasets')
plt.ylabel('AUC')
plt.show()


# Embedding Dimension VS AUC
axes = plt.gca()
x = [8,16,32,64]
y1 = [0.86, 0.8, 0.84, 0.84]
y2 = [0.69, 0.68, 0.70, 0]
y3 = [0.81, 0.85, 0.85, 0.82]

xi = [i for i in range(0,len(x))]
plt.plot(xi, y1, label = 'Haggle')
plt.plot(xi, y2, label = 'Hep-th')
plt.plot(xi, y3, label = 'MIT')
# axes.set_xlim([8,64])
axes.set_ylim([0,1])
plt.xticks(xi,x)

plt.title('Embedding Dimension V/S AUC')
plt.xlabel('Embedding Dimension')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Embedding Layer Count VS AUC
axes = plt.gca()
x = [3,4,5,6]
y1 = [0.89, 0.84, 0.88, 0.89]
y2 = [0.70, 0.70, 0.68, 0.70]
y3 = [0.86, 0.82, 0.80, 0.86]

xi = [i for i in range(0,len(x))]
plt.plot(xi, y1, label = 'Haggle')
plt.plot(xi, y2, label = 'Hep-th')
plt.plot(xi, y3, label = 'MIT')
# axes.set_xlim([8,64])
axes.set_ylim([0,1])
plt.xticks(xi,x)

plt.title('Number of Encoding Layers V/S AUC')
plt.xlabel('Number of Encoding Layers')
plt.ylabel('AUC')
plt.legend()
plt.show()


# Embedding Number Of Snapshots VS AUC
axes = plt.gca()
x = [4,5,6,7]
y1 = [0.83, 0.91, 0.92, 0.81]
y2 = [0.70, 0.70, 0.68, 0.70]
y3 = [0.82, 0.80, 0.88, 0.86]

xi = [i for i in range(0,len(x))]
plt.plot(xi, y1, label = 'Haggle')
plt.plot(xi, y2, label = 'Hep-th')
plt.plot(xi, y3, label = 'MIT')

# axes.set_xlim([8,64])
axes.set_ylim([0,1])
plt.xticks(xi,x)

plt.title('Number of Snapshots V/S AUC')
plt.xlabel('Number of Snapshots')
plt.ylabel('AUC')
plt.legend()
plt.show()