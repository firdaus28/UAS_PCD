import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Langkah 1: Baca citra
image = cv2.imread(r'Downloads\image.jpg')  # Ganti dengan path citra Anda
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konversi ke RGB

# Langkah 2: Reshape citra menjadi vektor fitur
# Setiap piksel direpresentasikan sebagai vektor [R, G, B]
pixel_values = image.reshape((-1, 3))  # Reshape menjadi (jumlah_piksel, 3)
pixel_values = np.float32(pixel_values)  # Konversi ke float32

# Langkah 3: Tentukan jumlah cluster (k)
k = 3  # Misalnya, kita ingin membagi citra menjadi 3 cluster

# Langkah 4: Terapkan K-Means Clustering
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixel_values)

# Langkah 5: Dapatkan label dan centroid
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Langkah 6: Ubah setiap piksel ke nilai centroid cluster-nya
segmented_image = centroids[labels].reshape(image.shape)  # Reshape kembali ke bentuk citra asli
segmented_image = np.uint8(segmented_image)  # Konversi ke tipe data uint8

# Langkah 7: Tampilkan hasil segmentasi
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Citra Asli')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Segmentasi dengan K-Means (k={k})')
plt.imshow(segmented_image)
plt.axis('off')

plt.show()