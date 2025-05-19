import numpy as np
from PIL import Image
import os


class CNN:
    def __init__(self):
        self.filter1 = np.random.randn(3, 3)
        self.filter2 = np.random.randn(3, 3)
        self.filter3 = np.random.randn(3, 3)
        self.filter4 = np.random.randn(3, 3)
        self.filter5 = np.random.randn(3, 3)
        self.filter6 = np.random.randn(3, 3)
        self.stride = 1
        self.padding = 0

    def prepImages(self):
        pngs = []
        current_dir = os.getcwd()
        save_dir = os.path.join(current_dir, "0")

        # Create save_dir if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Recursively find .jpg files
        for dirpath, _, filenames in os.walk(current_dir):
            for filename in filenames:
                if filename.lower().endswith(".jpg"):
                    file_path = os.path.join(dirpath, filename)
                    pngs.append(file_path)

        # Process and save
        for idx, p in enumerate(pngs):
            img = Image.open(p).convert("L")
            img = img.resize((28, 28))

            # Save with a unique filename
            new_filename = f"img_{idx}.jpg"
            self.save_path = os.path.join(save_dir, new_filename)
            img.save(self.save_path)

    def relu(self, z):
        return np.maximum(0, z)

    def convLayer1(self):
        self.image = Image.open("/home/fw7th/randomCode/3.jpg")
        image = np.array(self.image)
        image = np.array(image) / 255.0
        f_map1 = self.convolve2d(image, self.filter1)
        f_map2 = self.convolve2d(image, self.filter2)
        f_map3 = self.convolve2d(image, self.filter3)
        f_map4 = self.convolve2d(image, self.filter4)
        f_map5 = self.convolve2d(image, self.filter5)
        f_map6 = self.convolve2d(image, self.filter6)

        return f_map1, f_map2, f_map3, f_map4, f_map5, f_map6

    def convlayer2(self):
        # f_map1, f_map2, f_map3, f_map4, f_map5, f_map6 = self.convLayer1()
        pass

    def convolve2d(self, matrix, kernel):
        matrix_height = matrix.shape[0]
        matrix_width = matrix.shape[1]
        kernel_height = kernel.shape[0]
        kernel_width = kernel.shape[1]

        # General formula to calculate the output size
        output_height = (
            matrix_height - kernel_height + 2 * self.padding
        ) // self.stride + 1
        output_width = (
            matrix_width - kernel_width + 2 * self.padding
        ) // self.stride + 1

        padded_matrix = np.pad(matrix, self.padding, mode="constant")
        output_matrix = np.zeros((output_height, output_width))

        for i in range(
            0, matrix_height - kernel_height + 2 * self.padding + 1, self.stride
        ):
            for j in range(
                0, matrix_width - kernel_width + 2 * self.padding + 1, self.stride
            ):
                output_matrix[i // self.stride, j // self.stride] = np.sum(
                    padded_matrix[i : i + kernel_height, j : j + kernel_width] * kernel
                )

        return output_matrix

    def avgPooling(self, feature_map):
        f_height, f_width = feature_map.shape
        # vec_space =


# Example Usage
matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

cnn = CNN()

f1, f2, f3, f4, f5, f6 = cnn.convLayer1()
print(f1.shape)

cnn.avgPooling(f1)
