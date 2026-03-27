import cv2
from sklearn.cluster import KMeans
import numpy as np

# threshold: stops clustering if inertia doesn't improve by at least this much in percentage
def kmeans(image_bgr, threshold=0.20, iterations=1):
    best = {"inertia": None, "centroids": None, "kmeans": None, "iteration": None}
    for i in range(iterations):
        print("Iteration", i + 1)
        prev_inertia = None
        prev_centroids = None
        prev_kmeans = None
        k = 1
        data = image_bgr
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
        iteration_inertia = prev_inertia
        iteration_centroids = prev_centroids
        iteration_kmeans = prev_kmeans
        if best["inertia"] == None or iteration_inertia < best["inertia"]:
            best["inertia"] = iteration_inertia
            best["centroids"] = iteration_centroids
            best["kmeans"] = iteration_kmeans
            best["iteration"] = i + 1
    # we arent using this anymore, so it's safe to overwrite
    print(f'Using iteration {best["iteration"]} ({best["inertia"]})')
    centroids = (best["centroids"] * 255).round().astype(np.uint8)
    print(centroids)
    labels = best["kmeans"].predict(data_flat).reshape(data.shape[0:2])

    return centroids, labels