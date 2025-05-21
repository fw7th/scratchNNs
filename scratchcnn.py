import numpy as np
from PIL import Image


class CNN:
    def __init__(self):
        self.kernel2d = np.random.random(size=(6, 3, 3))
        self.kernel3d1 = np.random.random(size=(16, 6, 3, 3))
        self.kernel3d2 = np.random.random(size=(16, 16, 5, 5))
        self.bias2d = np.random.uniform(0.01, 2, size=(6, 1))
        self.bias3d1 = np.random.uniform(0.01, 2, size=(16, 1))
        self.bias3d2 = np.random.uniform(0.01, 2, size=(16, 1))
        self.fcbias = None
        self.stride = 1
        self.padding = 0

    def prepImages(self):
        pass

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        activation = 1 / (1 + np.exp(-z))
        return activation

    def convLayer1(self):
        self.image = Image.open("/home/fw7th/randomCode/3.jpg")
        feature_tensor = np.zeros((6, 26, 26))
        image = np.array(self.image)
        image = np.array(image) / 255.0
        for i in range(self.kernel2d.shape[0]):
            feat_map = self.convolve2d(image, self.kernel2d[i, :, :], self.bias2d[i])
            relu_map = self.relu(feat_map)
            feature_tensor[i, :, :] = relu_map

        return feature_tensor

    def poolLayer1(self):
        f_tensor = self.convLayer1()
        max_tensor = self.maxPool(f_tensor)

        return max_tensor

    def convLayer2(self):
        feat1 = self.poolLayer1()
        feature_tensor = np.zeros((16, 11, 11))

        for i in range(self.kernel3d1.shape[0]):
            feat_map = self.convolve3d(
                feat1, self.kernel3d1[i, :, :, :], self.bias3d1[i]
            )
            relu_map = self.relu(feat_map)
            feature_tensor[i, :, :] = relu_map

        return feature_tensor

    def poolLayer2(self):
        feat_tensor = self.convLayer2()
        max_layer = self.maxPool(feat_tensor)

        return max_layer

    def FCLayer1(self):
        max_layer = self.poolLayer2()
        flattened_layer = max_layer.flatten()
        flattened_layer = flattened_layer.reshape(flattened_layer.size, 1)
        self.f1 = np.zeros((flattened_layer.shape[0], 1))
        self.f1[:] = flattened_layer
        self.fcbias1 = np.random.uniform(0.01, 2, size=(10, 1))
        self.fcweights1 = np.random.randn(10, flattened_layer.size)
        self.f1_z = (self.fcweights1 @ self.f1) + self.fcbias1
        self.f1_activation = self.relu(self.f1_z)

        return self.f1_activation

    def FCLayer2(self):
        self.f2 = self.FCLayer1()
        self.fcbias2 = np.random.uniform(0.01, 2, size=(2, 1))
        self.fcweights2 = np.random.randn(2, 10)
        self.f2_z = (self.fcweights2 @ self.f2) + self.fcbias2
        self.f2_activation = self.relu(self.f2_z)

        return self.f2_activation

    def CNNOutput(self):
        f2_activation = self.FCLayer2()
        output = self.sigmoid(f2_activation)
        return output

    def convolve2d(self, matrix, kernel, bias):
        mat_h, mat_w = matrix.shape
        ker_h, ker_w = kernel.shape

        output_height = ((mat_h - ker_h + 2 * self.padding) // self.stride) + 1
        output_width = ((mat_w - ker_w + 2 * self.padding) // self.stride) + 1

        pad_mat = np.pad(matrix, self.padding, mode="constant")
        output_mat = np.zeros((output_height, output_width))

        for out_i, i in enumerate(range(0, (mat_h - ker_h) + 1, self.stride)):
            for out_j, j in enumerate(range(0, (mat_w - ker_w) + 1, self.stride)):
                sum = np.sum((pad_mat[i : i + ker_h, j : j + ker_w]) * kernel)
                output_mat[out_i, out_j] = sum + bias

        return output_mat

    def maxPool(self, feature_map):
        f_d, f_h, f_w = feature_map.shape
        ker_h, ker_w = 2, 2
        stride = 2

        pool_height = ((f_h - ker_h + 2 * self.padding) // stride) + 1
        pool_width = ((f_w - ker_w + 2 * self.padding) // stride) + 1

        pool = np.zeros((f_d, pool_height, pool_width))

        out_i = 0
        for i in range(0, (f_h - ker_h) + 1, stride):
            out_j = 0
            for j in range(0, (f_w - ker_h) + 1, stride):
                pool[:, out_i, out_j] = np.max(
                    feature_map[:, i : i + ker_h, j : j + ker_w], axis=(1, 2)
                )
                out_j += 1
            out_i += 1

        return pool

    def convolve3d(self, f_map, kernel, bias):
        f_depth, f_height, f_width = f_map.shape
        ker_depth, ker_height, ker_width = kernel.shape

        assert f_depth == ker_depth, "Depth mismatch between input and kernel"
        assert self.stride == 1, "This version only supports stride = 1"
        assert self.padding == 0, "This version assumes no padding"

        output_height = ((f_height - ker_height + 2 * self.padding) // self.stride) + 1
        output_width = ((f_width - ker_width + 2 * self.padding) // self.stride) + 1

        output = np.zeros((output_height, output_width))

        out_i = 0
        for i in range(0, (f_height - ker_height) + 1, self.stride):
            out_j = 0
            for j in range(0, (f_width - ker_width) + 1, self.stride):
                region = f_map[:, i : i + ker_height, j : j + ker_width]
                output[out_i, out_j] = np.sum(region * kernel) + bias
                out_j += 1
            out_i += 1

        return output


# Example Usage
cnn = CNN()
fc = cnn.FCLayer2()
print(fc.shape)
print(fc)
