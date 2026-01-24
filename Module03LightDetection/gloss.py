from kmeans import kmeans
import numpy as np

# convenience function
def cluster_gloss(image, clustering_threshold=0.2, clustering_iterations=1):
    centroids, labels = kmeans(image, clustering_threshold, clustering_iterations)
    return detect_gloss(centroids, labels)

def detect_gloss(centroids, labels):
    centroids = centroids.astype(np.uint16)
    gloss = (centroids ** 2).sum((1)).argmax()
    print("Gloss is centroid", gloss)
    for i in range(len(centroids)):
        print("[{:d}] {:0.2f}%".format(i, np.mean(labels == i) * 100))
    return np.mean(labels == gloss), gloss