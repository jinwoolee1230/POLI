// Copyright (c) 2020, Massachusetts Institute of Technology,
// Cambridge, MA 02139
// All Rights Reserved
// Authors: Jingnan Shi, et al. (see THANKS for the full author list)
// See LICENSE for the license information

#pragma once

#include <map>

#include "mex.h"

#include <Eigen/Core>

// Credit to Effective Modern C++ Item 10
template <typename E> constexpr typename std::underlying_type<E>::type toUType(E e) noexcept {
  return static_cast<typename std::underlying_type<E>::type>(e);
};

/**
 * Templated function to check if input is a R-by-C MATLAB matrix
 * @tparam R rows
 * @tparam C columns
 * @param pa
 * @return
 */
template <int R, int C> bool isRealDoubleMatrixAndCheckDimension(const mxArray* pa) {
  bool isDoubleMatrix = mxIsDouble(pa) && (!mxIsComplex(pa)) && (!mxIsScalar(pa));
  bool correctDimensions = (mxGetM(pa) == R) && (mxGetN(pa) == C);
  return isDoubleMatrix && correctDimensions;
}

/**
 * Return true if input is a 3-by-N MATLAB matrix
 * @param pa
 * @return
 */
bool isPointCloudMatrix(const mxArray* pa) {
  size_t rows = 3;
  bool isDoubleMatrix = mxIsDouble(pa) && (!mxIsComplex(pa)) && (!mxIsScalar(pa));
  bool correctDimensions = mxGetM(pa) == rows;
  return isDoubleMatrix && correctDimensions;
}

/**
 * @brief Return true if input is a N-by-N matrix
 * @param pa
 * @return
 */
bool isSquareMatrix(const mxArray* pa) {
  bool isDoubleMatrix = mxIsDouble(pa) && (!mxIsComplex(pa)) && (!mxIsScalar(pa));
  bool correctDimensions = mxGetM(pa) == mxGetN(pa);
  return isDoubleMatrix && correctDimensions;
}

/**
 * @brief Return true if input is a N-by-M matrix
 * @param pa
 * @return
 */
bool isRealDoubleMatrix(const mxArray* pa) {
  bool isDoubleMatrix = mxIsDouble(pa) && (!mxIsComplex(pa)) && (!mxIsScalar(pa));
  return isDoubleMatrix;
}

/**
 * Return true if input is a real double scalar
 * @param pa
 * @return
 */
bool isRealDoubleScalar(const mxArray* pa) {
  return mxIsDouble(pa) && mxIsScalar(pa) && (!mxIsComplex(pa));
}

/**
 * Convert a 3-by-N mxArray to Eigen 3-by-N matrix
 * @param pa
 */
void mexPointMatrixToEigenMatrix(const mxArray* pa,
                                 Eigen::Matrix<double, 3, Eigen::Dynamic>* eigen_matrix) {
  int rows = mxGetM(pa);
  int cols = mxGetN(pa);
  mexPrintf("row: %d cols: %d \n", rows, cols);
  if (rows != 3)
    return;
  eigen_matrix->resize(rows, cols);

  double* in_matrix;
  in_matrix = mxGetPr(pa);

  *eigen_matrix = Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic>>(in_matrix, rows, cols);
}

/**
 * Convert a 6-by-N mxArray to Eigen 6-by-N matrix
 * @param pa
 */
void mexPointNormalMatrixToEigenMatrix(const mxArray* pa,
                                       Eigen::Matrix<double, 6, Eigen::Dynamic>* eigen_matrix) {
  int rows = mxGetM(pa);
  int cols = mxGetN(pa);
  mexPrintf("row: %d cols: %d \n", rows, cols);
  if (rows != 6)
    return;
  eigen_matrix->resize(rows, cols);

  double* in_matrix;
  in_matrix = mxGetPr(pa);

  *eigen_matrix = Eigen::Map<Eigen::Matrix<double, 6, Eigen::Dynamic>>(in_matrix, rows, cols);
}

/**
 * Convert a N-by-M mxArray to Eigen 3-by-N matrix
 * @param pa
 */
void mexMatrixToEigenMatrix(const mxArray* pa,
                            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>* eigen_matrix) {
  int rows = mxGetM(pa);
  int cols = mxGetN(pa);
  eigen_matrix->resize(rows, cols);

  double* in_matrix;
  in_matrix = mxGetPr(pa);

  *eigen_matrix =
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(in_matrix, rows, cols);
}

/**
 * @brief Load a 3-by-N mxArray to a vector of 3-by-3 Eigen matrices
 * @param pa
 * @return
 */
std::vector<Eigen::Matrix3d> mexRotAvgMatrixToMatrixVector(const mxArray* pa) {
  int rows = mxGetM(pa);
  int cols = mxGetN(pa);
  assert(rows == 3);

  double* in_matrix;
  in_matrix = mxGetPr(pa);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> measurement_mat =
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(in_matrix, rows, cols);

  assert(cols % 3 == 0);
  size_t N = cols / 3;
  std::vector<Eigen::Matrix3d> measurement_vec;
  for (size_t i = 0; i < N; ++i) {
    size_t start_idx = 3 * i;
    measurement_vec.emplace_back(measurement_mat.block<3, 3>(0, start_idx));
  }
  return measurement_vec;
}
