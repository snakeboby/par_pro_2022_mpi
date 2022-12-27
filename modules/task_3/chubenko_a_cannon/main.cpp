// Copyright 2022 Chubenko Andrey
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "./chubenko_a_cannon.h"
#include "gtest-mpi-listener.hpp"

TEST(Parallel_Cannon_matrix_multiplication, Test_random_small) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> mtx1, mtx2, res, res_seq;
    const int count_size_mtx = 7;

    if (rank == 0) {
        mtx1 = getRandomSquareMatrix(count_size_mtx);
        mtx2 = getRandomSquareMatrix(count_size_mtx);
        res_seq = getSequentialMultiplication(mtx1, mtx2, count_size_mtx);
    }

    res = getParallelCannonMultiplication(mtx1, mtx2, count_size_mtx);

    if (rank == 0) {
        ASSERT_TRUE(matrixEqual(res, res_seq, count_size_mtx, 0.00001));
    }
}

TEST(Parallel_Cannon_matrix_multiplication, Test_random_medium) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> mtx1, mtx2, res, res_seq;
    const int count_size_mtx = 30;

    if (rank == 0) {
        mtx1 = getRandomSquareMatrix(count_size_mtx);
        mtx2 = getRandomSquareMatrix(count_size_mtx);
        res_seq = getSequentialMultiplication(mtx1, mtx2, count_size_mtx);
    }

    res = getParallelCannonMultiplication(mtx1, mtx2, count_size_mtx);

    if (rank == 0) {
        ASSERT_TRUE(matrixEqual(res, res_seq, count_size_mtx, 0.000001));
    }
}

TEST(Parallel_Cannon_matrix_multiplication, Test_large) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> mtx1, mtx2, res, res_seq;
    const int count_size_mtx = 100;

    if (rank == 0) {
        mtx1 = getRandomSquareMatrix(count_size_mtx);
        mtx2 = getRandomSquareMatrix(count_size_mtx);
        res_seq = getSequentialMultiplication(mtx1, mtx2, count_size_mtx);
    }

    res = getParallelCannonMultiplication(mtx1, mtx2, count_size_mtx);

    if (rank == 0) {
        ASSERT_TRUE(matrixEqual(res, res_seq, count_size_mtx, 0.0001));
    }
}

TEST(Parallel_Cannon_matrix_multiplication, Test_scalar) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> mtx1, mtx2, res, res_seq;
    const int count_size_mtx = 1;

    if (rank == 0) {
        mtx1 = getRandomSquareMatrix(count_size_mtx);
        mtx2 = getRandomSquareMatrix(count_size_mtx);
        res_seq = getSequentialMultiplication(mtx1, mtx2, count_size_mtx);
    }

    res = getParallelCannonMultiplication(mtx1, mtx2, count_size_mtx);

    if (rank == 0) {
        ASSERT_TRUE(matrixEqual(res, res_seq, count_size_mtx, 0.000001));
    }
}

TEST(Parallel_Cannon_matrix_multiplication, Test_is_right) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> mtx1, mtx2, res, res_seq, res_expected;
    const int count_size_mtx = 3;

    if (rank == 0) {
        mtx1 = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
        mtx2 = { 2.3, 7.5, 1.2, 3.4, 5.6, 9.8, 4.5, 6.7, 8.9 };
        res_expected = { 12.4, 19, 27.6, 43, 78.4, 87.3, 73.6, 137.8, 147 };
        res_seq = getSequentialMultiplication(mtx1, mtx2, count_size_mtx);
    }

    res = getParallelCannonMultiplication(mtx1, mtx2, count_size_mtx);

    if (rank == 0) {
        ASSERT_TRUE(matrixEqual(res, res_seq, count_size_mtx, 0.000001));
        ASSERT_TRUE(matrixEqual(res, res_expected, count_size_mtx, 0.000001));
    }
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
