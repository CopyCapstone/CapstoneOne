import cv2
from sklearn.cluster import KMeans
import numpy as np

# threshold: stops clustering if inertia doesn't improve by at least this much in percentage
def kmeans(image, threshold=0.20):
    prev_inertia = None
    prev_centroids = None
    prev_kmeans = None
    k = 1
    data = cv2.imread(image)
    data_flat = data.reshape((-1, 3))
    data_flat = data_flat / 255
    while True:
        print(f"Trying {k} cluster(s)... ", end="")
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_flat)
        centroids = kmeans.cluster_centers_
        inertia = kmeans.inertia_
        ratio = inertia / prev_inertia if k > 1 else 0
        print(kmeans.inertia_, "({:0.2f})".format(ratio))
        if ratio <= 1 - threshold:
            k += 1
            prev_centroids = centroids
            prev_inertia = inertia
            prev_kmeans = kmeans
        else:
            break
    # we arent using this anymore, so it's safe to overwrite
    centroids = (prev_centroids * 255).round().astype(np.uint8)
    print(centroids)
    labels = prev_kmeans.predict(data_flat).reshape(data.shape[0:2])

    return centroids, labels