// Copyright 2022 Chubenko Andrey
#ifndef MODULES_TASK_3_CHUBENKO_A_CANNON_CHUBENKO_A_CANNON_H_
#define MODULES_TASK_3_CHUBENKO_A_CANNON_CHUBENKO_A_CANNON_H_

#include <vector>
#include <string>

std::vector<double> getRandomSquareMatrix(int sz);
bool matrixEqual(std::vector<double> mtx1, std::vector<double> mtx2,
                    int sz, double tolerance);
std::vector<double> getParallelCannonMultiplication(std::vector<double> mtx1,
                    std::vector<double> mtx2, int sz);
std::vector<double> getSequentialMultiplication(std::vector<double> mtx1,
                    std::vector<double> mtx2, int sz);

#endif  // MODULES_TASK_3_CHUBENKO_A_CANNON_CHUBENKO_A_CANNON_H_
