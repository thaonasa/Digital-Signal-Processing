import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Hàm để tạo bộ lọc lý tưởng (Ideal) - Vuông


def ideal_lowpass_filter(size, cutoff):
    rows, cols = size
    center = (int(rows / 2), int(cols / 2))
    filter_matrix = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if distance <= cutoff:
                filter_matrix[i, j] = 1
    return filter_matrix

# Hàm để tạo bộ lọc tròn


def circular_lowpass_filter(size, cutoff):
    rows, cols = size
    center = (int(rows / 2), int(cols / 2))
    filter_matrix = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            filter_matrix[i, j] = 1 / (1 + (distance / cutoff) ** 2)
    return filter_matrix

# Hàm để tạo bộ lọc Gaussian


def gaussian_lowpass_filter(size, cutoff):
    rows, cols = size
    center = (int(rows / 2), int(cols / 2))
    filter_matrix = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            filter_matrix[i,
                          j] = np.exp(-(distance ** 2) / (2 * (cutoff ** 2)))
    return filter_matrix

# Hàm để tạo bộ lọc Butterworth


def butterworth_lowpass_filter(size, cutoff, order=2):
    rows, cols = size
    center = (int(rows / 2), int(cols / 2))
    filter_matrix = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            filter_matrix[i, j] = 1 / (1 + (distance / cutoff) ** (2 * order))
    return filter_matrix

# Hàm để hiển thị đáp ứng của các bộ lọc 2D và 3D


def plot_filter_response(filter_matrix, title):
    plt.figure(figsize=(12, 5))

    # Hiển thị đáp ứng 2D
    plt.subplot(1, 2, 1)
    plt.imshow(filter_matrix, cmap='gray')
    plt.title(f'2D Response - {title}')
    plt.colorbar()

    # Hiển thị đáp ứng 3D
    ax = plt.subplot(1, 2, 2, projection='3d')
    X, Y = np.meshgrid(
        range(filter_matrix.shape[1]), range(filter_matrix.shape[0]))
    ax.plot_surface(X, Y, filter_matrix, cmap='viridis')
    ax.set_title(f'3D Response - {title}')
    plt.show()

# Thực hiện lọc ảnh


def apply_filter(image, filter_matrix):
    # Chuyển đổi ảnh sang domain tần số
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # Áp dụng bộ lọc
    filtered_dft = dft_shift * filter_matrix

    # Chuyển ngược lại domain không gian
    inverse_dft = np.fft.ifftshift(filtered_dft)
    filtered_image = np.fft.ifft2(inverse_dft)
    filtered_image = np.abs(filtered_image)

    return filtered_image


# Đọc ảnh đầu vào từ file đã tải lên
# Đường dẫn đến file bạn đã tải lên
image_path = 'D:\Cá nhân\MSE\Xử lý tín hiệu số\ASM2-NguyenNgocThao\Fig0417(a)(barbara).png'
image = cv2.imread(image_path, 0)  # Đọc ảnh ở chế độ grayscale
if image is None:
    print("Error: Could not load image. Please check the file path.")
else:
    image_size = image.shape

    # Khởi tạo và hiển thị các bộ lọc
    cutoff_frequency = 30

    ideal_filter = ideal_lowpass_filter(image_size, cutoff_frequency)
    circular_filter = circular_lowpass_filter(image_size, cutoff_frequency)
    gaussian_filter = gaussian_lowpass_filter(image_size, cutoff_frequency)
    butterworth_filter = butterworth_lowpass_filter(
        image_size, cutoff_frequency)

    plot_filter_response(ideal_filter, 'Ideal Lowpass Filter')
    plot_filter_response(circular_filter, 'Circular Lowpass Filter')
    plot_filter_response(gaussian_filter, 'Gaussian Lowpass Filter')
    plot_filter_response(butterworth_filter, 'Butterworth Lowpass Filter')

    # Lọc ảnh với các bộ lọc
    filtered_image_ideal = apply_filter(image, ideal_filter)
    filtered_image_circular = apply_filter(image, circular_filter)
    filtered_image_gaussian = apply_filter(image, gaussian_filter)
    filtered_image_butterworth = apply_filter(image, butterworth_filter)

    # Hiển thị kết quả
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(filtered_image_ideal, cmap='gray')
    plt.title('Filtered Image - Ideal Lowpass')

    plt.subplot(2, 2, 2)
    plt.imshow(filtered_image_circular, cmap='gray')
    plt.title('Filtered Image - Circular Lowpass')

    plt.subplot(2, 2, 3)
    plt.imshow(filtered_image_gaussian, cmap='gray')
    plt.title('Filtered Image - Gaussian Lowpass')

    plt.subplot(2, 2, 4)
    plt.imshow(filtered_image_butterworth, cmap='gray')
    plt.title('Filtered Image - Butterworth Lowpass')

    plt.show()
