# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import math
import numpy as np

class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        sqrt_neg_one = complex(0, 1)

        R, C = matrix.shape
        Forward_matrix = np.zeros((R, C), dtype=complex)

        for u in range(R):
            for v in range(C):
                for i in range(R):
                    for j in range(C):
                        cosine = math.cos((2 * math.pi / R) * ((u * i) + (v * j)))
                        sine = sqrt_neg_one * math.sin((2 * math.pi / C) * ((u * i) + (v * j)))
                        Forward_matrix[u][v] = Forward_matrix[u][v] + (matrix[i][j] * (cosine - sine))

        return Forward_matrix

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""

        sqrt_neg_one = complex(0, 1)

        R, C = matrix.shape
        Inverse_matrix = np.zeros((R, C), dtype=complex)

        for u in range(R):
            for v in range(C):
                for i in range(R):
                    for j in range(C):
                        cosine = math.cos((2 * math.pi / R) * (u * i + v * j))
                        sine = sqrt_neg_one * math.sin((2 * math.pi / C) * (u * i + v * j))
                        Inverse_matrix[u][v] = Inverse_matrix[u][v] + (matrix[i][j] * (cosine + sine))

        return Inverse_matrix

    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""
        R, C = matrix.shape
        new_mat = np.zeros((R, C))
        for u in range(R):
            for v in range(C):
                sumPIX = 0
                for i in range(R):
                    for j in range(C):
                        sumPIX += matrix[i][j] * math.cos(((2 * math.pi) / R) * ((u * i) + (v * j)))
                new_mat[u][v] = sumPIX

        matrix = new_mat

        return matrix


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""
        R, C = matrix.shape
        new_mat = np.zeros((R, C), dtype=complex)
        r, c = matrix.shape
        for u in range(r):
            for v in range(c):
                sumPIX = 0
                "////////////////////"
                for i in range(R):
                    for j in range(C):
                        sumPIX += matrix[i][j] * (math.cos(((2 * math.pi) / R) * ((u * i) + (v * j))) - (1j * math.sin(((2 * math.pi) / R) * ((u * i) + (v * j)))))
                new_mat[u][v] = sumPIX
        "taking the absolute value without abs"
        for u in range(r):
            for v in range(c):
                if new_mat[u][v] < 0:
                    new_mat[u][v] = new_mat[u][v] * -1
        matrix = new_mat

        return matrix
