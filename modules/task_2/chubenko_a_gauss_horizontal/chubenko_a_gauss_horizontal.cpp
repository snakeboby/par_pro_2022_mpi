// Copyright 2022 Chubenko Andrey
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include "../../../modules/task_2/chubenko_a_gauss_horizontal/chubenko_a_gauss_horizontal.h"

std::vector<double> getRandomVector(int sz) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> dis(-5, 5);
    std::vector<double> vec(sz);
    for (int i = 0; i < sz; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

bool matrixEqual(const std::vector<double>& mtx1,
                 const std::vector<double>& mtx2, int sz, double tolerance) {
    for (int i = 0; i < sz * sz; i++) {
        if (abs(mtx1[i] - mtx2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

std::vector<double> solve_gaussian_method(std::vector<double> matrix,
                            std::vector<double> rhs, int block_size, int n) {
    // For each row i in the matrix, find the pivot element (largest)
    for (int i = 0; i < block_size; i++) {
        int pivot_row = i;
        double pivot_value = fabs(matrix[i * n + i]);
        for (int j = i + 1; j < block_size; j++) {
            double value = fabs(matrix[j * n + i]);
            if (value > pivot_value) {
                pivot_row = j;
                pivot_value = value;
            }
        }

        // Swap the pivot row with the current row
        if (pivot_row != i) {
            for (int j = 0; j < n; j++) {
                double temp = matrix[i * n + j];
                matrix[i * n + j] = matrix[pivot_row * n + j];
                matrix[pivot_row * n + j] = temp;
            }
            double temp = rhs[i];
            rhs[i] = rhs[pivot_row];
            rhs[pivot_row] = temp;
        }

        // For each row j below the pivot row, subtract a multiple of
        // the pivot row from row j so that the pivot element in row j
        // becomes zero
        for (int j = i + 1; j < block_size; j++) {
            double factor = matrix[j * n + i] / matrix[i * n + i];
            for (int k = i; k < n; k++) {
                matrix[j * n + k] -= factor * matrix[i * n + k];
            }
            rhs[j] -= factor * rhs[i];
        }
    }

    // Back-substitute to find the solution to the system of linear equations
    for (int i = block_size - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            double factor = matrix[j * n + i] / matrix[i * n + i];
            matrix[j * n + i] -= factor * matrix[i * n + i];
            rhs[j] -= factor * rhs[i];
        }
        rhs[i] /= matrix[i * n + i];
    }

    return rhs;
}

// Function to compute the Gaussian filtered image using a horizontal scheme
std::vector<double> horizontalGaussianMethod(
            const std::vector<double>& global_matrix,
            const std::vector<double>& global_rhs) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = global_rhs.size();
    int p = size;
    int block_size = n / p;

    // Allocate memory for the local matrix and right-hand side vector
    std::vector<double> local_matrix = std::vector<double>(block_size * n);
    std::vector<double> local_rhs = std::vector<double>(block_size);
    std::vector<double> local_solution = std::vector<double>(block_size);

    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < n; j++) {
            local_matrix[i * n + j] =
                global_matrix[(rank * block_size + i) * n + j];
        }
        local_rhs[i] = global_rhs[rank * block_size + i];
    }

    // Solve the system of linear equations using the Gaussian method
    local_solution = solve_gaussian_method(local_matrix, local_rhs,
                                            block_size, n);

    // Gather the solution vectors from all processes and
    // combine them into a single solution vector
    std::vector<double> global_solution = std::vector<double>(n);
    MPI_Gather(&local_solution[0], block_size, MPI_DOUBLE,
            &global_solution[0], block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return global_solution;
}
