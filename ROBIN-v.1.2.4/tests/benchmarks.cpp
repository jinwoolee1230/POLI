// Copyright (c) 2020, Massachusetts Institute of Technology,
// Cambridge, MA 02139
// All Rights Reserved
// Authors: Jingnan Shi, et al. (see THANKS for the full author list)
// See LICENSE for the license information

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <omp.h>
#include <set>

#include <Eigen/Core>

#include <robin/core.hpp>
#include <robin/graph.hpp>
#include <robin/problems.hpp>
#include <robin/robin.hpp>
#include <robin/utils.hpp>

/**
 * @brief Helper function to generate random 3D registration problems
 * @param T 4x4 transformation matrix
 * @return
 */
std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd>
GenerateRandom3dRegProblem(const Eigen::Matrix4d& T, const size_t& N, double noise_bound = 0.1) {

  Eigen::Matrix<double, 3, Eigen::Dynamic> src_points =
      Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
  Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
  src_h.resize(4, src_points.cols());
  src_h.topRows(3) = src_points;
  src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

  // apply transformation
  Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
  Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_points = tgt_h.topRows(3);

  // add noise
  Eigen::Matrix<double, 3, Eigen::Dynamic> noise =
      Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N) * noise_bound;
  tgt_points = tgt_points + noise;

  std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd> result{src_points, tgt_points};
  return result;
}

TEST_CASE("graph construction benchmark") {

  // an arbitrary transformation matrix
  Eigen::Matrix4d T;
  // clang-format off
  T << 9.96926560e-01,  6.68735757e-02, -4.06664421e-02, -1.15576939e-01,
      -6.61289946e-02, 9.97617877e-01,  1.94008687e-02, -3.87705398e-02,
      4.18675510e-02, -1.66517807e-02,  9.98977765e-01, 1.14874890e-01,
      0,              0,                0,              1;
  // clang-format on

  // generate a random 3d registration problem
  srand(1);
  double noise_bound = 0.1;
  auto problem = GenerateRandom3dRegProblem(T, 1000, noise_bound);

  // setup the graph constructor
  // Measurement 3D points set
  robin::Points3d measurements(problem.second);

  // Model set
  // Note: for our purpose measurements and model points can be flipped
  robin::Points3d model(problem.first);

  // Compatibility function
  robin::Points3dRegCompCheck comp_check(&model, noise_bound);

  // InvGraphConstructor
  robin::CompGraphConstructor<robin::Points3d, robin::Points3dRegCompCheck, 2> graph_constructor;
  graph_constructor.SetCompCheckFunction(&comp_check);
  graph_constructor.SetMeasurements(&measurements);

  // serial
  BENCHMARK_ADVANCED("serial")
  (Catch::Benchmark::Chronometer meter) {
    // measure
    auto function_to_test = [&]() {
      robin::AdjListGraph g;
      graph_constructor.BuildCompGraph_serial(&g);
      return g;
    };

    meter.measure([&] { return function_to_test(); });
  };

  // parallel adj edge buffer
  BENCHMARK_ADVANCED("parallel adj edge buffer")
  (Catch::Benchmark::Chronometer meter) {
    // measure
    auto function_to_test = [&]() {
      robin::AdjListGraph g;
      graph_constructor.BuildCompGraph_parallel_edge_buffer(&g);
      return g;
    };

    meter.measure([&] { return function_to_test(); });
  };

  // parallel adj vertex parallel
  BENCHMARK_ADVANCED("parallel adj vertex parallel")
  (Catch::Benchmark::Chronometer meter) {
    // measure
    auto function_to_test = [&]() {
      robin::AdjListGraph g;
      graph_constructor.BuildCompGraph_parallel_vertex(&g);
      return g;
    };

    meter.measure([&] { return function_to_test(); });
  };

  // parallel csr two passes
  BENCHMARK_ADVANCED("parallel csr two passes")
  (Catch::Benchmark::Chronometer meter) {
    // measure
    auto function_to_test = [&]() {
      robin::CSRGraph g;
      graph_constructor.BuildCSRCompGraph_parallel_two_passes(&g);
      return g;
    };

    meter.measure([&] { return function_to_test(); });
  };

  // parallel csr two passes atomic
  BENCHMARK_ADVANCED("parallel csr two passes atomic")
  (Catch::Benchmark::Chronometer meter) {
    // measure
    auto function_to_test = [&]() {
      robin::AtomicCSRGraph g;
      graph_constructor.BuildCSRCompGraph_parallel_two_passes_atomic(&g);
      return g;
    };

    meter.measure([&] { return function_to_test(); });
  };
}

TEST_CASE("graph construction csr two passes atomic benchmark") {

  // an arbitrary transformation matrix
  Eigen::Matrix4d T;
  // clang-format off
  T << 9.96926560e-01,  6.68735757e-02, -4.06664421e-02, -1.15576939e-01,
      -6.61289946e-02, 9.97617877e-01,  1.94008687e-02, -3.87705398e-02,
      4.18675510e-02, -1.66517807e-02,  9.98977765e-01, 1.14874890e-01,
      0,              0,                0,              1;
  // clang-format on

  // generate a random 3d registration problem
  srand(1);
  double noise_bound = 0.1;
  auto problem = GenerateRandom3dRegProblem(T, 500, noise_bound);

  // setup the graph constructor
  // Measurement 3D points set
  robin::Points3d measurements(problem.second);

  // Model set
  // Note: for our purpose measurements and model points can be flipped
  robin::Points3d model(problem.first);

  // Compatibility function
  robin::Points3dRegCompCheck comp_check(&model, noise_bound);

  // InvGraphConstructor
  robin::CompGraphConstructor<robin::Points3d, robin::Points3dRegCompCheck, 2> graph_constructor;
  graph_constructor.SetCompCheckFunction(&comp_check);
  graph_constructor.SetMeasurements(&measurements);

  // parallel csr two passes atomic
  BENCHMARK_ADVANCED("parallel csr two passes atomic")
  (Catch::Benchmark::Chronometer meter) {
    // measure
    auto function_to_test = [&]() {
      robin::AtomicCSRGraph g;
      graph_constructor.BuildCSRCompGraph_parallel_two_passes_atomic(&g);
      return g;
    };

    meter.measure([&] { return function_to_test(); });
  };
}
