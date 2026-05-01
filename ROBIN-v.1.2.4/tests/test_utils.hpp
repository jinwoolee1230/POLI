// Copyright (c) 2020, Massachusetts Institute of Technology,
// Cambridge, MA 02139
// All Rights Reserved
// Authors: Jingnan Shi, et al. (see THANKS for the full author list)
// See LICENSE for the license information

#pragma once

#include "catch.hpp"

#include <cstdlib>

#include <Eigen/Core>

#include <robin/core.hpp>
#include <robin/graph.hpp>
#include <robin/problems.hpp>
#include <robin/robin.hpp>
#include <robin/utils.hpp>

namespace robin {
namespace test {

/**
 * @brief Fixture for vector averaging problems
 */
class VecAveragingProblemFixture {
public:
  VecAveragingProblemFixture() = default;

  Eigen::Matrix2Xd GenerateMeasurements(const Eigen::Vector2d& exp_vector, size_t N,
                                        size_t outlier_count = 0, double noise_bound = 0.1) {
    assert(outlier_count <= N);
    // noises between -0.1,0.1
    Eigen::Matrix2Xd random_noises = Eigen::Matrix2Xd::Random(2, N);
    random_noises.colwise().normalize();
    random_noises *= noise_bound;

    // manually create outliers
    if (outlier_count > 0) {
      for (size_t i = 0; i < outlier_count; ++i) {
        random_noises.col(i) *= 10 * (1 + rand() % 10);
        ExpectedOutliers.push_back(i);
      }
    }

    Eigen::Matrix2Xd measurements = Eigen::Matrix2Xd::Zero(2, N);
    for (size_t i = 0; i < random_noises.cols(); ++i) {
      measurements.col(i) = exp_vector + random_noises.col(i);
    }

    return measurements;
  }

  std::vector<size_t> ExpectedOutliers;
};

} // namespace test

} // namespace robin