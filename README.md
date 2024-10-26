# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Convert emails into numerical features using tokenization, lowercasing, and TF-IDF or Bag
 of Words.
 2. Transform the processed text into feature vectors for SVM input.
 3. Train an SVM classifier (with a linear or other kernel) on labeled data to distinguish
 between spam and not spam emails.
 4. Use the trained SVM model to predict whether new emails are spam and evaluate
 performance using metrics like accuracy and precision

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by: Sreeviveka V.S 
RegisterNumber:  2305001031  
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data

#Extract features
X=data[['Annual Income (k$)','Spending Score (1-100)']]

plt.figure(figsize=(4,4))
plt.scatter(X['Annual Income (k$)'],X['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

k=3
Kmeans=KMeans(n_clusters=k)
Kmeans.fit(X)
centroids=Kmeans.cluster_centers_


labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)

colors=['r','g','b']
for i in range(k):
  cluster_points=X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:

![WhatsApp Image 2024-10-26 at 15 26 26_7d6a2b3d](https://github.com/user-attachments/assets/d8ca428e-ee5d-498e-9a74-faa8a34ded0d)
![WhatsApp Image 2024-10-26 at 15 26 32_d3759c88](https://github.com/user-attachments/assets/2e9ac484-2336-4bf5-8b05-8febc2bc1b85)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
