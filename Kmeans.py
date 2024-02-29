import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

data = pd.read_csv('Customer data.csv', skiprows=1, header=None) #Loading Data from CSV

data = data.iloc[:, 1:].values #Skipping the first column

max_clusters = 50  # Determine the range of clusters to try
inertia_values = []

for num_clusters in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    inertia_values.append(kmeans.inertia_)

#Plot the elbow curve
plt.plot(range(1, max_clusters + 1), inertia_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Within-cluster sum of squares)')
#plt.figure()

#From the previous elbow curve function we can deduce that the elbow is located around K = 7 clusters

K = 7 #Number of clusters

def GUC_Distance(Cluster_Centroids, data, Distance_Type):
        if Distance_Type == 'Euclidean':
            Cluster_Distance = np.zeros((len(data), len(Cluster_Centroids)))
            for i, centroid in enumerate(Cluster_Centroids):
                centroid = centroid.reshape(1, -1)  # Reshape to match the number of features in data
            if centroid.shape[1] < data.shape[1]:
                centroid = np.pad(centroid, ((0, 0), (0, data.shape[1] - centroid.shape[1])), 'constant')
            Cluster_Distance[:, i] = np.linalg.norm(data - centroid, axis=1)
        elif Distance_Type == 'Pearson':
            Mean = np.mean(data, axis=0)
            std = np.std(data, axis=0, ddof=1) #Standard Deviation
            for i in range (len(std)):
                if(std[i] == 0): #So it doesnt divide by 0
                    std[i] = 1e-9
                norm = (data-Mean) / std  #Standardization of the data
                C_norm = (Cluster_Centroids - Mean) / std #Standardization of the cluster centroids
                Cluster_Distance = 1 - np.dot(norm,C_norm.T)
        else:
            return "Please Enter a Valid Distance Type"
        return Cluster_Distance

def GUC_Kmean(data, k, Distance_Type):
    # Generating the Centroids randomly
    Clusters = data[random.sample(range(len(data)), k), 1:].astype(int)
    # Getting the distance between the heads and the data
    Final_Cluster_Distance = GUC_Distance(Clusters, data, Distance_Type)

    # Checking for minimum distance and index for each cluster
    while True:
        Cluster_Assignment = np.argmin(Final_Cluster_Distance, axis=1)  # Getting the smallest value in each row
        old_Clusters = np.copy(Clusters)

        # Update Centroids
        for i in range(k):
            Cluster_points = data[Cluster_Assignment == i][:, 1:].astype(float)
            if len(Cluster_points) > 0:
                Clusters[i, :] = np.nanmean(Cluster_points, axis=0)
            else:
                Clusters[i, :] = np.zeros_like(Clusters[0])

        Final_Cluster_Distance = GUC_Distance(Clusters, data, Distance_Type)  # Recalculating the distance
        Cluster_Metric = np.sum(np.min(Final_Cluster_Distance, axis=1) ** 2)  # Distortion function of the last iteration only will be returned
        print("hello")
        # Stopping Condition
        if np.array_equal(old_Clusters, Clusters):
            break

    return Cluster_Assignment, Final_Cluster_Distance,Cluster_Metric

   
#HelperDisplayFucntionOriginally
def display_cluster(X, km=[], num_clusters=0):
    color = 'brgcmyk'  # List colors
    alpha = 0.5  # Color opacity
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c=color[0],alpha=alpha,s=s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_==i,0],X[km.labels_==i,1],c=color[i],alpha=alpha, s=s)
            plt.scatter(km.cluster_centers_[i][0], km.cluster_centers_[i][1], c=color[i], marker='x', s=100)

#Example_1
plt.rcParams['figure.figsize'] = [8,8]
sns.set_style("whitegrid")
sns.set_context("talk")
angle = np.linspace(0,2*np.pi,20, endpoint = False)
X_Circle = np.append([np.cos(angle)],[np.sin(angle)],0).transpose()
display_cluster(X_Circle)
#plt.figure()

#Example_2
n_samples = 1000
n_bins = 4  
centers = [(-3, -3), (0, 0), (3, 3), (6, 6), (9,9)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)
kmeans = KMeans(n_clusters=len(centers), random_state=42)
kmeans.fit(X)
display_cluster(X, km=kmeans, num_clusters=len(centers))
#plt.figure()

#Example_3
n_samples = 1000
X, y = noisy_moons = make_moons(n_samples=n_samples, noise= .1)
display_cluster(X)
#plt.show()

#Updated Helper Function
def display_cluster_updated(X, labels=[], num_clusters=0):
    color = 'brgcmyk'  # List colors
    alpha = 0.5  # Color opacity
    s = 20
    if X.shape[1] > 2:
        num_dimensions = X.shape[1]
        fig, axis = plt.subplots(nrows=num_dimensions-1, ncols=num_dimensions-1, figsize=(12, 12))
        axis = axis.flatten()

        for dim1 in range(num_dimensions-1):
            for dim2 in range(dim1+1, num_dimensions):
                ax = axis[dim1 * (num_dimensions-1) + dim2 - 1]
                ax.set_title(f'Dimensions {dim1+1} vs {dim2+1}')
                if num_clusters == 0:
                    ax.scatter(X[:, dim1], X[:, dim2], c=color[0], alpha=alpha, s=s)
                else:
                    for i in range(num_clusters):
                        ax.scatter(X[labels==i, dim1], X[labels==i, dim2], c=color[i], alpha=alpha, s=s)
                        ax.scatter(km.cluster_centers_[i][dim1], km.cluster_centers_[i][dim2], c=color[i], marker='x', s=100)

        plt.tight_layout()
    else:
        if num_clusters == 0:
            plt.scatter(X[:, 0], X[:, 1], c=color[0], alpha=alpha, s=s)
        else:
            for i in range(num_clusters):
                plt.scatter(X[labels==i, 0], X[labels==i, 1], c=color[i], alpha=alpha, s=s)
               
                plt.scatter(km.cluster_centers_[i][0], km.cluster_centers_[i][1], c=color[i], marker='x', s=100)


Cluster_Assignment, Final_Cluster_Distance,Cluster_Metric= GUC_Kmean(data, K, 'Euclidean')

display_cluster_updated(X, labels=Cluster_Assignment, num_clusters=K)
plt.show()


