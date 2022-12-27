// Copyright 2022 Chubenko Andrey
#include <mpi.h>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include "../../../modules/task_3/chubenko_a_cannon.h"


std::vector<double> getRandomSquareMatrix(int sz) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> dis(-5, 5);
    std::vector<double> mtx(sz * sz);
    for (int i = 0; i < sz * sz; i++) {
        mtx[i] = dis(gen);
    }
    return mtx;
}

bool matrixEqual(std::vector<double> mtx1, std::vector<double> mtx2, int sz,
                    double tolerance) {
    for (int i = 0; i < sz * sz; i++) {
        if (abs(mtx1[i] - mtx2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Simple square matrix multiplication
std::vector<double> getSequentialMultiplication(std::vector<double> mtx1,
                                        std::vector<double> mtx2, int sz) {
    std::vector<double> res(sz * sz);
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            res[i * sz + j] = 0;
            for (int k = 0; k < sz; k++) {
                res[i * sz + j] += mtx1[i * sz + k] * mtx2[k * sz + j];
            }
        }
    }
    return res;
}

// Cannon algorithm for square matrix multiplication
std::vector<double> getParallelCannonMultiplication(std::vector<double> mtx1,
                                            std::vector<double> mtx2, int sz) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Calculate the block size and the size of the 2D grid of processes
    int sqrt_size = static_cast<int>(sqrt(size));
    int block_sz = sz / sqrt_size;

    // Allocate space for the result matrix and the local blocks
    std::vector<double> res(sz * sz);
    std::vector<double> block1(block_sz * block_sz);
    std::vector<double> block2(block_sz * block_sz);
    std::vector<double> block_res(block_sz * block_sz);

    // Create MPI datatypes for the blocks
    MPI_Datatype block;
    MPI_Type_vector(block_sz, block_sz, sz, MPI_DOUBLE, &block);
    MPI_Type_commit(&block);
    MPI_Datatype block_col;
    MPI_Type_vector(block_sz, 1, block_sz, MPI_DOUBLE, &block_col);
    MPI_Type_commit(&block_col);
    MPI_Datatype block_row;
    MPI_Type_vector(1, block_sz, sz, MPI_DOUBLE, &block_row);
    MPI_Type_commit(&block_row);

    // Create a 2D cartesian communicator
    int shift_source, shift_dest;
    int coords[2];
    int periods[2] = { 0, 0 };
    MPI_Comm comm_2d;
    int dims[2] = { sqrt_size, sqrt_size };
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_2d);
    MPI_Cart_coords(comm_2d, rank, 2, coords);

    // Scatter the matrix blocks to the processes
    if (rank == 0) {
        for (int i = 0; i < block_sz; i++) {
            for (int j = 0; j < block_sz; j++) {
                block1[i * block_sz + j] = mtx1[i * sz + j];
                block2[i * block_sz + j] = mtx2[i * sz + j];
            }
        }
    }
    MPI_Scatter(&mtx1[0], 1, block, &block1[0],
                block_sz * block_sz, MPI_DOUBLE, 0, comm_2d);
    MPI_Scatter(&mtx2[0], 1, block, &block2[0],
                block_sz * block_sz, MPI_DOUBLE, 0, comm_2d);

    // Perform the multiplication
    for (int i = 0; i < sqrt_size; i++) {
        shift_source = (coords[0] + i) % sqrt_size;
        shift_dest = (coords[1] + i) % sqrt_size;
        MPI_Sendrecv_replace(&block1[0], block_sz * block_sz, MPI_DOUBLE,
                            shift_dest, 0, shift_source, 0, comm_2d,
                            MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(&block2[0], block_sz * block_sz, MPI_DOUBLE,
                            shift_source, 0, shift_dest, 0, comm_2d,
                            MPI_STATUS_IGNORE);
        block_res = getSequentialMultiplication(block1, block2, block_sz);
        for (int k = 0; k < block_sz; k++) {
            for (int l = 0; l < block_sz; l++) {
                res[k * sz + l] += block_res[k * block_sz + l];
            }
        }
    }

    // Gather the result matrix blocks
    MPI_Gather(&block_res[0], 1, block, &res[0],
                block_sz * block_sz, MPI_DOUBLE, 0, comm_2d);
    return res;
}
