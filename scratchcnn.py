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
        self.kernel3d = np.random.randn(self.f_depth, 3, 3)
        self.bias1 = np.random.uniform(0.1, 2)
        self.bias2 = np.random.uniform(0.1, 2)
        self.bias3 = np.random.uniform(0.1, 2)
        self.bias4 = np.random.uniform(0.1, 2)
        self.bias5 = np.random.uniform(0.1, 2)
        self.bias6 = np.random.uniform(0.1, 2)
        self.bias3d = np.random.uniform(0.1, 2)
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
        f_map1 = self.relu(self.convolve2d(image, self.filter1, self.bias1))
        f_map2 = self.relu(self.convolve2d(image, self.filter2, self.bias2))
        f_map3 = self.relu(self.convolve2d(image, self.filter3, self.bias3))
        f_map4 = self.relu(self.convolve2d(image, self.filter4, self.bias4))
        f_map5 = self.relu(self.convolve2d(image, self.filter5, self.bias5))
        f_map6 = self.relu(self.convolve2d(image, self.filter6, self.bias6))

        return f_map1, f_map2, f_map3, f_map4, f_map5, f_map6

    def poolLayer1(self):
        f1, f2, f3, f4, f5, f6 = self.convLayer1()
        f1_pool = self.maxPool(f1)
        f2_pool = self.maxPool(f2)
        f3_pool = self.maxPool(f3)
        f4_pool = self.maxPool(f4)
        f5_pool = self.maxPool(f5)
        f6_pool = self.maxPool(f6)

        f_stack1 = np.stack(
            (f1_pool, f2_pool, f3_pool, f4_pool, f5_pool, f6_pool), axis=0
        )
        return f_stack1

    def convlayer2(self):
        f_stack1 = self.poolLayer1()
        pass

    def convolve2d(self, matrix, kernel, bias):
        mat_h, mat_w = matrix.shape
        ker_h, ker_w = kernel.shape

        output_height = ((mat_h - ker_h + 2 * self.padding) // self.stride) + 1
        output_width = ((mat_w - ker_w + 2 * self.padding) // self.stride) + 1

        pad_mat = np.pad(matrix, self.padding, mode="constant")
        output_mat = np.zeros((output_height, output_width))

        for i in range(0, output_height, self.stride):
            for j in range(0, output_width, self.stride):
                sum = np.sum((pad_mat[i : i + ker_h, j : j + ker_w]) * kernel)
                output_mat[i // self.stride, j // self.stride] = sum + bias

        return output_mat

    def maxPool(self, feature_map):
        f_h, f_w = feature_map.shape
        ker_h, ker_w = 2, 2

        pool_height = ((f_h - ker_h + 2 * self.padding) // self.stride) + 1
        pool_width = ((f_w - ker_w + 2 * self.padding) // self.stride) + 1

        pool = np.zeros((pool_height, pool_width))

        for i in range(0, pool_height, self.stride):
            for j in range(0, pool_width, self.stride):
                pool[i, j] = np.max(
                    feature_map[i : i + pool_height, j : j + pool_width]
                )

        return pool

    def convolve3d(self, f_map):
        self.f_depth, f_height, f_width = f_map.shape
        ker_depth, ker_height, ker_width = self.kernel3d.shape

        output_height = ((f_height - ker_height + 2 * self.padding) // self.stride) + 1
        output_width = ((f_width - ker_width + 2 * self.padding) // self.stride) + 1

        output_matrix = np.zeros((output_height, output_width))

        for i in range(0):
            for j in range(0, f_height, self.stride):
                for k in range(0, f_width, self.stride):
                    total_value = (
                        np.sum(
                            f_map[i, j : j + ker_height, k : k + ker_width]
                            * self.kernel3d[i + 1, :, :]
                        )
                        + np.sum(
                            f_map[i + 1, j : j + ker_height, k : k + ker_width]
                            * self.kernel3d[i + 1, :, :]
                        )
                        + np.sum(
                            f_map[i + 2, j : j + ker_height, k : k + ker_width]
                            * self.kernel3d[i + 2, :, :]
                        )
                        + np.sum(
                            f_map[i + 3, j : j + ker_height, k : k + ker_width]
                            * self.kernel3d[i + 3, :, :]
                        )
                        + np.sum(
                            f_map[i + 4, j : j + ker_height, k : k + ker_width]
                            * self.kernel3d[i + 4, :, :]
                        )
                        + np.sum(
                            f_map[i + 5, j : j + ker_height, k : k + ker_width]
                            * self.kernel3d[i + 5, :, :]
                        )
                    )

                    output_matrix[i, j] = total_value + self.bias3d

        return output_matrix


# Example Usage
matrix = np.array(
    [
        [1, 2, 3, 4, 5],
        [5, 5, 6, 7, 8],
        [4, 9, 10, 11, 12],
        [2, 6, 3, 4, 5],
        [5, 6, 5, 3, 2],
    ]
)
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

cnn = CNN()
"""
f1, f2, f3, f4, f5, f6 = cnn.convLayer1()
print(f1.shape)
"""
cow = cnn.convolve2d(matrix, kernel, 0.234)
p = cnn.maxPool(cow)
print(cow.shape)
print(p.shape)

print(f"{cow}\n \n {p} \n \n")
cran = cnn.convLayer1()
print(cran)
