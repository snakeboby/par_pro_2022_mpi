// Copyright 2022 Chubenko Andrey
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "./chubenko_a_gauss_horizontal.h"
#include <gtest-mpi-listener.hpp>


TEST(Gauss_Horizontal, Test_random_7) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> global_matrix;
  const int count_size_vector = 7;

  if (rank == 0) {
    global_matrix =
        getRandomVector(count_size_vector * (count_size_vector + 1));
  }

  std::vector<double> global_solution =
      horizontalGaussianMethod(global_matrix, count_size_vector);

  if (rank == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      double sum = 0;
      for (int j = 0; j < count_size_vector; j++) {
        sum +=
            global_matrix[i * (count_size_vector + 1) + j] * global_solution[j];
      }

      ASSERT_NEAR(
          global_matrix[i * (count_size_vector + 1) + count_size_vector], sum,
          1e-9);
    }
  }
}

TEST(Gauss_Horizontal, Test_random_30) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> global_matrix;
  const int count_size_vector = 30;

  if (rank == 0) {
    global_matrix =
        getRandomVector(count_size_vector * (count_size_vector + 1));
  }

  std::vector<double> global_solution =
      horizontalGaussianMethod(global_matrix, count_size_vector);

  if (rank == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      double sum = 0;
      for (int j = 0; j < count_size_vector; j++) {
        sum +=
            global_matrix[i * (count_size_vector + 1) + j] * global_solution[j];
      }

      ASSERT_NEAR(
          global_matrix[i * (count_size_vector + 1) + count_size_vector], sum,
          1e-9);
    }
  }
}

TEST(Gauss_Horizontal, Test_random_100) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> global_matrix;
  const int count_size_vector = 100;

  if (rank == 0) {
    global_matrix =
        getRandomVector(count_size_vector * (count_size_vector + 1));
  }

  std::vector<double> global_solution =
      horizontalGaussianMethod(global_matrix, count_size_vector);

  if (rank == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      double sum = 0;
      for (int j = 0; j < count_size_vector; j++) {
        sum +=
            global_matrix[i * (count_size_vector + 1) + j] * global_solution[j];
      }

      ASSERT_NEAR(
          global_matrix[i * (count_size_vector + 1) + count_size_vector], sum,
          1e-9);
    }
  }
}

TEST(Gauss_Horizontal, Test_random_1) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> global_matrix;
  const int count_size_vector = 1;

  if (rank == 0) {
    global_matrix =
        getRandomVector(count_size_vector * (count_size_vector + 1));
  }

  std::vector<double> global_solution =
      horizontalGaussianMethod(global_matrix, count_size_vector);

  if (rank == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      double sum = 0;
      for (int j = 0; j < count_size_vector; j++) {
        sum +=
            global_matrix[i * (count_size_vector + 1) + j] * global_solution[j];
      }

      ASSERT_NEAR(
          global_matrix[i * (count_size_vector + 1) + count_size_vector], sum,
          1e-9);
    }
  }
}

//
TEST(Gauss_Horizontal, Test_right) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<double> global_matrix, ref_solution;
  const int count_size_vector = 3;

  if (rank == 0) {
    global_matrix = {1, 1, 1, 3, 2, 3, 7, 0, 1, 3, -2, 17};
    ref_solution = {1, 4, -2};
  }

  std::vector<double> global_solution =
      horizontalGaussianMethod(global_matrix, count_size_vector);

  if (rank == 0) {
    ASSERT_EQ(ref_solution, global_solution);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners &listeners =
      ::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}
