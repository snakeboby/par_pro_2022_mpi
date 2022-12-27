// Copyright 2022 Chubenko Andrey
#ifndef MODULES_TASK_2_CHUBENKO_A_GAUSS_HORIZONTAL_CHUBENKO_A_GAUSS_HORIZONTAL_H_
#define MODULES_TASK_2_CHUBENKO_A_GAUSS_HORIZONTAL_CHUBENKO_A_GAUSS_HORIZONTAL_H_

#include <vector>
#include <string>

std::vector<double> getRandomVector(int  sz);
bool matrixEqual(const std::vector<double>& mtx1,
                 const std::vector<double>& mtx2, int sz, double tolerance);
std::vector<double> solve_gaussian_method(
                std::vector<double> matrix,
                std::vector<double> rhs, int block_size, int n);
std::vector<double> horizontalGaussianMethod(
                const std::vector<double>& global_matrix,
                const std::vector<double>& global_rhs);

#endif  // MODULES_TASK_2_CHUBENKO_A_GAUSS_HORIZONTAL_CHUBENKO_A_GAUSS_HORIZONTAL_H_
