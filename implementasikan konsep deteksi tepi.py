import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Fungsi untuk mengaplikasikan operator Roberts
def roberts_operator(image):
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    # Konvolusi dengan kernel Roberts
    gradient_x = convolve2d(image, kernel_x, mode='same')
    gradient_y = convolve2d(image, kernel_y, mode='same')
    
    # Menggabungkan gradien
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude

# Fungsi untuk mengaplikasikan operator Sobel
def sobel_operator(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # Konvolusi dengan kernel Sobel
    gradient_x = convolve2d(image, kernel_x, mode='same')
    gradient_y = convolve2d(image, kernel_y, mode='same')
    
    # Menggabungkan gradien
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude

# Fungsi untuk melakukan konvolusi 2D
def convolve2d(image, kernel, mode='same'):
    return np.absolute(np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(kernel, s=image.shape))).astype(np.float32)

# Membaca citra
image = imageio.imread(r'Downloads\image.jpg', mode='F')  # Menggunakan mode='F' untuk grayscale

# Mengaplikasikan operator Roberts
roberts_edge = roberts_operator(image)

# Mengaplikasikan operator Sobel
sobel_edge = sobel_operator(image)

# Menampilkan hasil
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Citra Asli')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(roberts_edge, cmap='gray')
plt.title('Deteksi Tepi Roberts')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sobel_edge, cmap='gray')
plt.title('Deteksi Tepi Sobel')
plt.axis('off')

plt.show()