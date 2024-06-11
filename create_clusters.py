import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.applications as kapp
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import shutil

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = kapp.vgg16.preprocess_input(x)
    features = model.predict(x)
    features = np.squeeze(features)
    return features

vgg16_model = kapp.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

images_folder = "dataset"

image_files = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.endswith('.jpg')]

all_features = []
for img_file in tqdm(image_files, desc="Extracting Features"):
    features = extract_features(img_file, vgg16_model)
    all_features.append(features)

all_features = normalize(np.array(all_features))

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(all_features)
labels = kmeans.labels_

clusters_folder = os.path.join(os.path.dirname(images_folder), "dataset-clusters")
os.makedirs(clusters_folder, exist_ok=True)

for i in range(num_clusters):
    cluster_images = [image_files[j] for j in range(len(image_files)) if labels[j] == i]
    cluster_folder = os.path.join(clusters_folder, f"Cluster-{i+1}")
    os.makedirs(cluster_folder, exist_ok=True)
    for img_file in cluster_images:
        shutil.copy(img_file, cluster_folder)

    print(f"Cluster {i+1}: {len(cluster_images)} images")

print("Clustering and copying complete.")
