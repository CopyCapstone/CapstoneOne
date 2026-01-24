from kmeans import kmeans
import numpy as np

# convenience function
def cluster_gloss(image, clustering_threshold=0.2):
    centroids, labels = kmeans(image, clustering_threshold)
    return detect_gloss(centroids, labels)

def detect_gloss(centroids, labels):
    centroids = centroids.astype(np.uint16)
    gloss = (centroids ** 2).sum((1)).argmax()
    print("Gloss is centroid", gloss)
    for i in range(len(centroids)):
        print(np.mean(labels == i))
    return np.mean(labels == gloss), gloss