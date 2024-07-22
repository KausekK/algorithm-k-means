import math
import random


def initialize_centroids_random(data, k):
    indices = random.sample(range(len(data)), k)
    centroids = [data[index] for index in indices]
    return centroids


def convert_data(data):
    processed_data = []
    for line in data:
        parts = line.strip().split(';')
        if parts[-1] == '':
            parts.pop()
        float_features = [float(part.replace(',', '.')) for part in parts[:-1]]
        processed_data.append(float_features)
    return processed_data


def assign_to_nearest_centroid(data, centroids):
    if isinstance(data[0], list):
        clusters = []
        for point in data:
            closest_centroid_index = find_nearest_centroid(point, centroids)
            clusters.append(closest_centroid_index)
        return clusters
    else:
        return find_nearest_centroid(data, centroids)


def update_centroids(data, clusters, k):
    new_centroids = [[0] * len(data[0]) for _ in range(k)]
    counts = [0] * k

    for i in range(len(data)):
        cluster_index = clusters[i]
        counts[cluster_index] += 1
        for j in range(len(data[i])):
            new_centroids[cluster_index][j] += data[i][j]

    for i in range(k):
        if counts[i] > 0:
            new_centroids[i] = [x / counts[i] for x in new_centroids[i]]
        else:
            new_centroids[i] = random.choice(data)

    return new_centroids


def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids_random(data, k)
    for _ in range(max_iterations):
        clusters = assign_to_nearest_centroid(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if centroids == new_centroids:
            break
        centroids = new_centroids
    return centroids, clusters


def find_nearest_centroid(point, centroids):
    min_distance = float('inf')
    closest_centroid_index = 0
    for i, centroid in enumerate(centroids):
        distance = math.sqrt(sum((point[j] - centroid[j]) ** 2 for j in range(len(point))))
        if distance < min_distance:
            min_distance = distance
            closest_centroid_index = i
    return closest_centroid_index


with open('Iris.data', 'r') as iris_data:
    data = iris_data.read().splitlines()
data = convert_data(data)

try:
    k = int(input("Podaj liczbę klastrów: "))
    if k <= 0:
        raise ValueError("Liczba klastrów musi być większa od zera.")
except ValueError as e:
    print(f"Wystąpił błąd: {e}")
    exit()

final_centroids, final_clusters = k_means(data, k)

for id, cluster_id in enumerate(final_clusters):
    print(f"Wektor {id + 1} został przydzielony do klastra {cluster_id + 1}")

try:
    new_vector = input("Podaj nowy wektor danych (oddzielone średnikami): ")
    new_vector = [float(x.strip().replace(',', '.')) for x in new_vector.split(';')]
    cluster_assignment = assign_to_nearest_centroid(new_vector, final_centroids)
    print(f"Ten wektor należy do klastra: {cluster_assignment + 1}")
except Exception as e:
    print(f"Wystąpił błąd: {e}")

import matplotlib.pyplot as plt


def plot_clusters(data, clusters, centroids):
    colors = ['r', 'g', 'b']
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
    fig, ax = plt.subplots()
    for cluster_index in range(len(centroids)):
        cluster_points = [data[i] for i in range(len(data)) if clusters[i] == cluster_index]
        x = [point[0] for point in cluster_points]  # Sepal length
        y = [point[1] for point in cluster_points]  # Sepal width
        ax.scatter(x, y, color=colors[cluster_index], label=f'Cluster {cluster_index + 1}', alpha=0.5)
        ax.scatter(centroids[cluster_index][0], centroids[cluster_index][1], color=colors[cluster_index], marker='x',
                   s=100)

    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_title('Iris Clusters and Centroids')
    ax.legend()
    plt.show()


plot_clusters(data, final_clusters, final_centroids)
