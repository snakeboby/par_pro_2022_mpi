// Copyright 2022 Chubenko Andrey
#ifndef MODULES_TASK_2_CHUBENKO_A_GAUSS_HORIZONTAL_CHUBENKO_A_GAUSS_HORIZONTAL_H_
#define MODULES_TASK_2_CHUBENKO_A_GAUSS_HORIZONTAL_CHUBENKO_A_GAUSS_HORIZONTAL_H_

#include <vector>
#include <string>

std::vector<double> getRandomVector(int  sz);
std::vector<double> horizontalGaussianMethod(
    const std::vector<double>& global_matrix, const int n);


#endif  // MODULES_TASK_2_CHUBENKO_A_GAUSS_HORIZONTAL_CHUBENKO_A_GAUSS_HORIZONTAL_H_
