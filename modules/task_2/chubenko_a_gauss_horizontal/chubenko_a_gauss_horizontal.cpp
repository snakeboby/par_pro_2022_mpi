// Copyright 2022 Chubenko Andrey
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
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

// Get id of a global matrix row based on local
int get_row_by_rank_row(int rank, int local_id, int size) {
  return local_id * size + rank;
}

// Get rank based on global row id
int get_rank_by_row(int global_id, int size) { return global_id % size; }

std::vector<double>
horizontalGaussianMethod(const std::vector<double> &global_matrix,
                         const int n) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int local_size = n / size + (rank < n % size);
  // Array with pivot rows for each iteration
  std::vector<int> pivot_iter(n);
  std::vector<double> local_full_matrix(local_size * (n + 1));
  std::vector<double> local_matrix(local_size * n);
  std::vector<double> local_rhs(local_size);
  std::vector<double> local_solution(local_size);
  std::vector<int> global_solution_index(n);
  std::vector<double> global_solution(n);
  std::vector<double> pivot_row(n);

  // Fill local matrix and right hand side
  if (rank == 0) {
    int index = 0;
    for (int i = 0; i < n; i += size) {
      for (int j = 0; j < n; j++) {
        local_matrix[index * n + j] = global_matrix[i * (n + 1) + j];
      }
      local_rhs[index] = global_matrix[i * (n + 1) + n];
      index++;
    }
    for (int i = 1; i < size; i++) {
      int local_size = n / size + (i < n % size);
      std::vector<double> to_send(local_size * (n + 1));
      int index = 0;
      for (int j = i; j < n; j += size) {
        for (int k = 0; k < n; k++) {
          to_send[index * (n + 1) + k] = global_matrix[j * (n + 1) + k];
        }
        to_send[index * (n + 1) + n] = global_matrix[j * (n + 1) + n];
        index++;
      }
      MPI_Send(&to_send[0], local_size * (n + 1), MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD);
    }
  } else {
    MPI_Status status;
    MPI_Recv(&local_full_matrix[0], local_size * (n + 1), MPI_DOUBLE, 0, 0,
             MPI_COMM_WORLD, &status);
    for (int i = 0; i < local_size; i++) {
      for (int j = 0; j < n; j++) {
        local_matrix[i * n + j] = local_full_matrix[i * (n + 1) + j];
      }
    }
    for (int i = 0; i < local_size; i++) {
      local_rhs[i] = local_full_matrix[i * (n + 1) + n];
    }
  }

  // Parallel gaussian elimination

  std::vector<bool> used(local_size, false);

  // Forward loop
  for (int i = 0; i < n; i++) {
    // Get local pivot row (max)
    int global_pivot_row = -1;
    int local_pivot_row = -1;
    double local_pivot_value = 0;
    for (int j = 0; j < local_size; j++) {
      if (!used[j] && fabs(local_matrix[j * n + i]) > fabs(local_pivot_value)) {
        local_pivot_row = j;
        local_pivot_value = local_matrix[j * n + i];
      }
    }
    // Get global pivot row (max)
    struct {
      double abs_value;
      int rank;
    } local_pivot{}, global_pivot{};
    local_pivot.abs_value = fabs(local_pivot_value);
    if (local_pivot_row == -1) {
      local_pivot.abs_value = -1;
    }
    local_pivot.rank = rank;
    MPI_Allreduce(&local_pivot, &global_pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                  MPI_COMM_WORLD);

    // Broadcast pivot row id
    if (rank == global_pivot.rank) {
      global_pivot_row = get_row_by_rank_row(rank, local_pivot_row, size);
    }
    MPI_Bcast(&global_pivot_row, 1, MPI_INT, global_pivot.rank, MPI_COMM_WORLD);

    if (rank == 0) {
      pivot_iter[i] = global_pivot_row;
    }

    double pivot_b;
    if (rank == global_pivot.rank) {
      used[local_pivot_row] = true;
      for (int j = 0; j < n; j++) {
        pivot_row[j] = local_matrix[local_pivot_row * n + j];
      }
      pivot_b = local_rhs[local_pivot_row];
    }
    MPI_Bcast(&pivot_row[0], n, MPI_DOUBLE, global_pivot.rank, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, global_pivot.rank, MPI_COMM_WORLD);

    // Divide local rows by pivot row
    for (int j = 0; j < local_size; j++) {
      if (used[j]) {
        continue;
      }
      double ratio = local_matrix[j * n + i] / pivot_row[i];
      for (int k = i; k < n; k++) {
        local_matrix[j * n + k] -= ratio * pivot_row[k];
      }
      local_rhs[j] -= ratio * pivot_b;
    }
  }

  // Fill used with false
  for (int i = 0; i < local_size; i++) {
    used[i] = false;
  }

  // Backward loop
  for (int i = n - 1; i >= 0; i--) {
    int pivot_row_id = 0;
    if (rank == 0) {
      pivot_row_id = pivot_iter[i];
    }
    MPI_Bcast(&pivot_row_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int pivot_rank = get_rank_by_row(pivot_row_id, size);
    int pivot_local_row = pivot_row_id / size;

    double pivot_val;
    double pivot_b;
    if (rank == pivot_rank) {
      // Get local solution
      local_solution[pivot_local_row] =
          local_rhs[pivot_local_row] / local_matrix[pivot_local_row * n + i];
      pivot_val = local_matrix[pivot_local_row * n + i];
      pivot_b = local_rhs[pivot_local_row];
      used[pivot_local_row] = true;
    }
    if (rank == 0) {
      global_solution_index[pivot_row_id] = i;
    }
    MPI_Bcast(&pivot_val, 1, MPI_DOUBLE, pivot_rank, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, pivot_rank, MPI_COMM_WORLD);

    // Divide local rows by pivot row
    for (int j = 0; j < local_size; j++) {
      if (used[j]) {
        continue;
      }
      double ratio = local_matrix[j * n + i] / pivot_val;
      local_matrix[j * n + i] -= ratio * pivot_val;
      local_rhs[j] -= ratio * pivot_b;
    }
  }

  // Gather solution
  std::vector<double> to_recv;
  if (rank == 0) {
    for (int i = 0; i < local_size; i++) {
      global_solution[global_solution_index[get_row_by_rank_row(0, i, size)]] =
          local_solution[i];
    }
    for (int i = 1; i < size; i++) {
      int local_rank_size = n / size + (i < n % size);
      to_recv.resize(local_rank_size);
      MPI_Status status;
      MPI_Recv(&to_recv[0], local_rank_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
               &status);
      for (int j = 0; j < local_rank_size; j++) {
        global_solution[global_solution_index[get_row_by_rank_row(
            i, j, size)]] = to_recv[j];
      }
    }
  } else {
    MPI_Send(&local_solution[0], local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  return global_solution;
}
