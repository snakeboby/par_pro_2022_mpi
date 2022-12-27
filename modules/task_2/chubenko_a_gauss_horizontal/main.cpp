// Copyright 2022 Chubenko Andrey
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "./chubenko_a_gauss_horizontal.h"
#include <gtest-mpi-listener.hpp>

TEST(Gauss_Horizontal, Test_random_7) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> global_matrix, global_rhs, ref_solution;
    const int count_size_vector = 7;

    if (rank == 0) {
        global_matrix = getRandomVector(count_size_vector * count_size_vector);
        global_rhs = getRandomVector(count_size_vector);
    }

    std::vector<double> global_solution =
                horizontalGaussianMethod(global_matrix, global_rhs);

    if (rank == 0) {
        ref_solution = solve_gaussian_method(global_matrix, global_rhs,
                                        count_size_vector, count_size_vector);
        ASSERT_EQ(ref_solution, global_solution);
    }
}

TEST(Gauss_Horizontal, Test_random_30) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> global_matrix, global_rhs, ref_solution;
    const int count_size_vector = 30;

    if (rank == 0) {
        global_matrix = getRandomVector(count_size_vector * count_size_vector);
        global_rhs = getRandomVector(count_size_vector);
    }

    std::vector<double> global_solution =
                    horizontalGaussianMethod(global_matrix, global_rhs);

    if (rank == 0) {
        ref_solution = solve_gaussian_method(global_matrix, global_rhs,
                                        count_size_vector, count_size_vector);
        ASSERT_EQ(ref_solution, global_solution);
    }
}

TEST(Gauss_Horizontal, Test_random_100) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> global_matrix, global_rhs, ref_solution;
    const int count_size_vector = 100;

    if (rank == 0) {
        global_matrix = getRandomVector(count_size_vector * count_size_vector);
        global_rhs = getRandomVector(count_size_vector);
    }

    std::vector<double> global_solution =
                    horizontalGaussianMethod(global_matrix, global_rhs);

    if (rank == 0) {
        ref_solution = solve_gaussian_method(global_matrix, global_rhs,
                                        count_size_vector, count_size_vector);
        ASSERT_EQ(ref_solution, global_solution);
    }
}

TEST(Gauss_Horizontal, Test_random_1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> global_matrix, global_rhs, ref_solution;
    double res;
    const int count_size_vector = 1;

    if (rank == 0) {
        global_matrix = getRandomVector(count_size_vector * count_size_vector);
        global_rhs = getRandomVector(count_size_vector);
        res = global_rhs[0] / global_matrix[0];
    }

    std::vector<double> global_solution =
                    horizontalGaussianMethod(global_matrix, global_rhs);

    if (rank == 0) {
        ref_solution = solve_gaussian_method(global_matrix, global_rhs,
                                        count_size_vector, count_size_vector);
        ASSERT_EQ(ref_solution, global_solution);
        ASSERT_EQ(global_solution[0], res);
    }
}

TEST(Gauss_Horizontal, Test_right) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> global_matrix, global_rhs, ref_solution;
    const int count_size_vector = 3;

    if (rank == 0) {
        global_matrix = { 1, 1, 1, 2, 3, 7, 1, 3, -2 };
        global_rhs = { 3, 0, 17 };
        ref_solution = { 1, 4, -2 };
    }

    std::vector<double> global_solution =
                    horizontalGaussianMethod(global_matrix, global_rhs);

    if (rank == 0) {
        ASSERT_EQ(ref_solution, global_solution);
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
