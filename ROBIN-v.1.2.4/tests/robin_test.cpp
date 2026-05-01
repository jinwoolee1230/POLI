// Copyright (c) 2020, Massachusetts Institute of Technology,
// Cambridge, MA 02139
// All Rights Reserved
// Authors: Jingnan Shi, et al. (see THANKS for the full author list)
// See LICENSE for the license information

#include "catch.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <iterator>
#include <random>
#include <set>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <robin/robin.hpp>

TEST_CASE("vector averaging") {
  //
  // Prepare test data
  //
  size_t N = 10;
  size_t dimension = 5;
  Eigen::VectorXd exp_vec(dimension, 1);

  // noises between -0.1,0.1
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> random_noises =
      Eigen::MatrixXd::Random(dimension, N);
  random_noises.colwise().normalize();
  random_noises /= 10;
  double noise_bound = 0.1;

  // manually create outliers
  random_noises.col(0) *= 10;
  random_noises.col(9) *= 20;
  std::vector<size_t> expected_inliers = {1, 2, 3, 4, 5, 6, 7, 8};

  // actual measurements
  Eigen::MatrixXd measurements = Eigen::MatrixXd::Zero(dimension, N);
  for (size_t i = 0; i < random_noises.cols(); ++i) {
    measurements.col(i) = exp_vec + random_noises.col(i);
  }

  // call API
  auto* g = robin::MakeVecAvgInvGraph(measurements, 2 * noise_bound);
  auto actual_max_core_indices =
      robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CORE);
  std::sort(actual_max_core_indices.begin(), actual_max_core_indices.end());
  auto actual_max_clique_indices =
      robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
  std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());

  REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers));
  REQUIRE_THAT(actual_max_core_indices, Catch::Equals(expected_inliers));

  delete g;
}

TEST_CASE("vector averaging csr") {
  //
  // Prepare test data
  //
  size_t N = 10;
  size_t dimension = 5;
  Eigen::VectorXd exp_vec(dimension, 1);

  // noises between -0.1,0.1
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> random_noises =
      Eigen::MatrixXd::Random(dimension, N);
  random_noises.colwise().normalize();
  random_noises /= 10;
  double noise_bound = 0.1;

  // manually create outliers
  random_noises.col(0) *= 10;
  random_noises.col(9) *= 20;
  std::vector<size_t> expected_inliers = {1, 2, 3, 4, 5, 6, 7, 8};

  // actual measurements
  Eigen::MatrixXd measurements = Eigen::MatrixXd::Zero(dimension, N);
  for (size_t i = 0; i < random_noises.cols(); ++i) {
    measurements.col(i) = exp_vec + random_noises.col(i);
  }

  // call API
  auto* g = robin::MakeVecAvgInvGraph(measurements, 2 * noise_bound, robin::GraphsStorageType::CSR);
  auto actual_max_core_indices =
      robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CORE);
  std::sort(actual_max_core_indices.begin(), actual_max_core_indices.end());
  auto actual_max_clique_indices =
      robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
  std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());

  REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers));
  REQUIRE_THAT(actual_max_core_indices, Catch::Equals(expected_inliers));

  delete g;
}

TEST_CASE("vector averaging atomic csr") {
  //
  // Prepare test data
  //
  size_t N = 10;
  size_t dimension = 5;
  Eigen::VectorXd exp_vec(dimension, 1);

  // noises between -0.1,0.1
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> random_noises =
      Eigen::MatrixXd::Random(dimension, N);
  random_noises.colwise().normalize();
  random_noises /= 10;
  double noise_bound = 0.1;

  // manually create outliers
  random_noises.col(0) *= 10;
  random_noises.col(9) *= 20;
  std::vector<size_t> expected_inliers = {1, 2, 3, 4, 5, 6, 7, 8};

  // actual measurements
  Eigen::MatrixXd measurements = Eigen::MatrixXd::Zero(dimension, N);
  for (size_t i = 0; i < random_noises.cols(); ++i) {
    measurements.col(i) = exp_vec + random_noises.col(i);
  }

  // call API
  auto* g = robin::MakeVecAvgInvGraph(measurements, 2 * noise_bound,
                                      robin::GraphsStorageType::ATOMIC_CSR);
  auto actual_max_core_indices =
      robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CORE);
  std::sort(actual_max_core_indices.begin(), actual_max_core_indices.end());
  auto actual_max_clique_indices =
      robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
  std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());

  REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers));
  REQUIRE_THAT(actual_max_core_indices, Catch::Equals(expected_inliers));

  delete g;
}

TEST_CASE("vector averaging adj list") {
  //
  // Prepare test data
  //
  size_t N = 10;
  size_t dimension = 5;
  Eigen::VectorXd exp_vec(dimension, 1);

  // noises between -0.1,0.1
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> random_noises =
      Eigen::MatrixXd::Random(dimension, N);
  random_noises.colwise().normalize();
  random_noises /= 10;
  double noise_bound = 0.1;

  // manually create outliers
  random_noises.col(0) *= 10;
  random_noises.col(9) *= 20;
  std::vector<size_t> expected_inliers = {1, 2, 3, 4, 5, 6, 7, 8};

  // actual measurements
  Eigen::MatrixXd measurements = Eigen::MatrixXd::Zero(dimension, N);
  for (size_t i = 0; i < random_noises.cols(); ++i) {
    measurements.col(i) = exp_vec + random_noises.col(i);
  }

  // call API
  auto* g =
      robin::MakeVecAvgInvGraph(measurements, 2 * noise_bound, robin::GraphsStorageType::ADJ_LIST);
  auto actual_max_core_indices =
      robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CORE);
  std::sort(actual_max_core_indices.begin(), actual_max_core_indices.end());
  auto actual_max_clique_indices =
      robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
  std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());

  REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers));
  REQUIRE_THAT(actual_max_core_indices, Catch::Equals(expected_inliers));

  delete g;
}

TEST_CASE("single rotation averaging") {
  // generate a random rotation matrix
  Eigen::Quaternion<double> exp_mean_quat = Eigen::Quaternion<double>::UnitRandom();
  Eigen::Matrix3d exp_mean_mat = exp_mean_quat.normalized().toRotationMatrix();

  // noise parameters
  double noise_ceiling = 0.1; // radian
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, noise_ceiling);

  // generate SO(3) measurements with bounded noise
  size_t N = 10;
  std::vector<Eigen::Matrix3d> measurements;
  for (size_t i = 0; i < N; ++i) {
    double noise = dist(e2);
    Eigen::AngleAxis<double> random_pertubation =
        Eigen::AngleAxis<double>(noise, Eigen::Vector3d::UnitX());
    measurements.emplace_back(exp_mean_mat * random_pertubation.toRotationMatrix());
  }

  SECTION("no outliers") {
    // generate the graph
    auto* g = robin::MakeRotAvgInvGraph(measurements, robin::So3Distance::GEODESIC_DISTANCE,
                                        2 * noise_ceiling, robin::GraphsStorageType::ADJ_LIST);
    auto actual_max_core_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CORE);
    std::sort(actual_max_core_indices.begin(), actual_max_core_indices.end());
    auto actual_max_clique_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
    std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());

    // check results
    std::vector<size_t> expected_inliers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers));

    delete g;
  }

  SECTION("random outliers") {
    // add outliers
    measurements[0] = measurements[0] * Eigen::AngleAxis<double>((10 + dist(e2)) * noise_ceiling,
                                                                 Eigen::Vector3d::UnitY());
    measurements[3] = measurements[3] * Eigen::AngleAxis<double>((10 + dist(e2)) * noise_ceiling,
                                                                 Eigen::Vector3d::UnitZ());

    // expected inliers
    // measurement 0 & 3 are outliers
    std::vector<size_t> expected_inliers = {1, 2, 4, 5, 6, 7, 8, 9};

    // generate the graph
    auto* g = robin::MakeRotAvgInvGraph(measurements, robin::So3Distance::GEODESIC_DISTANCE,
                                        2 * noise_ceiling, robin::GraphsStorageType::ADJ_LIST);
    auto actual_max_core_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CORE);
    std::sort(actual_max_core_indices.begin(), actual_max_core_indices.end());
    auto actual_max_clique_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
    std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());

    // check results
    REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers));

    delete g;
  }
}

TEST_CASE("3d reg") {
  // an arbitrary transformation matrix
  Eigen::Matrix4d T;
  // clang-format off
  T << 9.96926560e-01,  6.68735757e-02, -4.06664421e-02, -1.15576939e-01,
      -6.61289946e-02, 9.97617877e-01,  1.94008687e-02, -3.87705398e-02,
      4.18675510e-02, -1.66517807e-02,  9.98977765e-01, 1.14874890e-01,
      0,              0,                0,              1;
  // clang-format on

  size_t N = 5;
  SECTION("no outliers") {

    // random 3d points
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points =
        Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
    Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
    src_h.resize(4, src_points.cols());
    src_h.topRows(3) = src_points;
    src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

    // apply transformation
    Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_points = tgt_h.topRows(3);

    // noise bound
    double noise_bound = 0.1;

    // add noise
    Eigen::Matrix<double, 3, Eigen::Dynamic> noise =
        Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N) * noise_bound;
    tgt_points = tgt_points + noise;

    // call robin API
    auto* g = robin::Make3dRegInvGraph(src_points, tgt_points, 2 * noise_bound,
                                       robin::GraphsStorageType::ADJ_LIST);
    // auto adj_mat = g.GetAdjMat();
    // std::cout << adj_mat << std::endl;

    // find the clique
    auto actual_max_core_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CORE);
    std::sort(actual_max_core_indices.begin(), actual_max_core_indices.end());
    auto actual_max_clique_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
    std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());

    // no outliers
    std::vector<size_t> expected_inliers = {0, 1, 2, 3, 4};
    REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers));

    delete g;
  }

  SECTION("random outliers") {

    // random 3d points
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points =
        Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
    Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
    src_h.resize(4, src_points.cols());
    src_h.topRows(3) = src_points;
    src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

    // apply transformation
    Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_points = tgt_h.topRows(3);

    // noise bound
    double noise_bound = 0.1;

    // add noise
    Eigen::Matrix<double, 3, Eigen::Dynamic> noise =
        Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N) * noise_bound;
    tgt_points = tgt_points + noise;

    // create an outlier at index 2
    tgt_points.col(2) *= 100;

    // call robin API
    auto* g = robin::Make3dRegInvGraph(src_points, tgt_points, 2 * noise_bound,
                                       robin::GraphsStorageType::ADJ_LIST);
    // auto adj_mat = g.GetAdjMat();
    // std::cout << adj_mat << std::endl;

    // find the clique
    auto actual_max_core_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CORE);
    std::sort(actual_max_core_indices.begin(), actual_max_core_indices.end());
    auto actual_max_clique_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
    std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());

    // outlier at index 2
    std::vector<size_t> expected_inliers = {0, 1, 3, 4};
    REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers));

    delete g;
  }
}

TEST_CASE("3d reg large instance") {
  // For benchmarking / profiling
  // an arbitrary transformation matrix
  Eigen::Matrix4d T;
  // clang-format off
  T << 9.96926560e-01,  6.68735757e-02, -4.06664421e-02, -1.15576939e-01,
      -6.61289946e-02, 9.97617877e-01,  1.94008687e-02, -3.87705398e-02,
      4.18675510e-02, -1.66517807e-02,  9.98977765e-01, 1.14874890e-01,
      0,              0,                0,              1;
  // clang-format on

  size_t N = 100;

  // random 3d points
  Eigen::Matrix<double, 3, Eigen::Dynamic> src_points =
      Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
  Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
  src_h.resize(4, src_points.cols());
  src_h.topRows(3) = src_points;
  src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

  // apply transformation
  Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
  Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_points = tgt_h.topRows(3);

  // noise bound
  double noise_bound = 0.1;

  // add noise
  Eigen::Matrix<double, 3, Eigen::Dynamic> noise =
      Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N) * noise_bound;
  tgt_points = tgt_points + noise;

  // call robin API
  auto* g = robin::Make3dRegInvGraph(src_points, tgt_points, 2 * noise_bound);

  // find the clique
  auto actual_max_core_indices =
      robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CORE);
  std::sort(actual_max_core_indices.begin(), actual_max_core_indices.end());
  auto actual_max_clique_indices =
      robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
  std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());

  delete g;
}

TEST_CASE("3d reg with normals") {
  // an arbitrary transformation matrix
  Eigen::Matrix4d T;
  // clang-format off
  T << 9.96926560e-01,  6.68735757e-02, -4.06664421e-02, -1.15576939e-01,
      -6.61289946e-02, 9.97617877e-01,  1.94008687e-02, -3.87705398e-02,
      4.18675510e-02, -1.66517807e-02,  9.98977765e-01, 1.14874890e-01,
      0,              0,                0,              1;
  // clang-format on

  SECTION("no outliers") {
    size_t N = 5;

    // random 3d points
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points =
        Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
    Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
    src_h.resize(4, src_points.cols());
    src_h.topRows(3) = src_points;
    src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

    // apply transformation
    Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_points = tgt_h.topRows(3);

    // populate normals
    Eigen::Matrix<double, 3, Eigen::Dynamic> normals(3, N);
    normals = Eigen::Matrix<double, 3, Eigen::Dynamic>::Ones(3, N);

    // populate correspondences
    Eigen::Matrix<double, 6, Eigen::Dynamic> src(6, N);
    src.topRows<3>() = src_points;
    src.bottomRows<3>() = normals;

    Eigen::Matrix<double, 6, Eigen::Dynamic> tgt(6, N);
    tgt.topRows<3>() = tgt_points;
    tgt.bottomRows<3>() = normals;

    std::vector<size_t> expected_inliers = {0, 1, 2, 3, 4};

    // noise bound
    Eigen::Vector2d noise_bound;
    noise_bound << 0.1, 0.1;

    // call robin API
    auto* g = robin::Make3dNormalRegInvGraph(src, tgt, noise_bound);

    // find the clique
    auto actual_max_core_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CORE);
    std::sort(actual_max_core_indices.begin(), actual_max_core_indices.end());
    auto actual_max_clique_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
    std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());

    REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers));

    delete g;
  }

  SECTION("outliers in normals") {
    //
    // This portion test the case where one of the normals is an outlier
    //
    size_t N = 5;

    // random 3d points
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points =
        Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
    Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
    src_h.resize(4, src_points.cols());
    src_h.topRows(3) = src_points;
    src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

    // apply transformation
    Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_points = tgt_h.topRows(3);

    // populate normals
    Eigen::Matrix<double, 3, Eigen::Dynamic> normals(3, N);
    normals = Eigen::Matrix<double, 3, Eigen::Dynamic>::Ones(3, N);

    // populate source
    Eigen::Matrix<double, 6, Eigen::Dynamic> src(6, N);
    src.topRows<3>() = src_points;
    src.bottomRows<3>() = normals;

    // add outlier (the corres. at index=3)
    size_t outlier_idx = 3;
    normals(0, outlier_idx) = 3;
    normals(1, outlier_idx) = 5;
    normals(2, outlier_idx) = 10;

    // populate target
    Eigen::Matrix<double, 6, Eigen::Dynamic> tgt(6, N);
    tgt.topRows<3>() = tgt_points;
    tgt.bottomRows<3>() = normals;

    double outlier_deviation_angle =
        std::acos((normals.col(outlier_idx).dot(normals.col(0))) /
                  (normals.col(outlier_idx).norm() * normals.col(0).norm()));

    // expected inliers
    std::vector<size_t> expected_inliers = {0, 1, 2, 4};

    // noise bound
    // with this noise bound, the outlier should be identified
    Eigen::Vector2d noise_bound;
    noise_bound << 0.1, std::abs(outlier_deviation_angle) * 0.1;

    // call robin API
    auto* g = robin::Make3dNormalRegInvGraph(src, tgt, noise_bound);

    // find the clique
    auto actual_max_core_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CORE);
    std::sort(actual_max_core_indices.begin(), actual_max_core_indices.end());
    auto actual_max_clique_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
    std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());

    REQUIRE_THAT(actual_max_core_indices, Catch::Equals(expected_inliers));
    REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers));

    // new (larger) noise bound
    // with this noise bound, the outlier should not be identified
    noise_bound << 0.1, std::abs(outlier_deviation_angle) * 1.5;

    // call robin API
    auto* g2 = robin::Make3dNormalRegInvGraph(src, tgt, noise_bound);

    // find the clique
    auto actual_max_core_indices_2 =
        robin::FindInlierStructure(g2, robin::InlierGraphStructure::MAX_CORE);
    std::sort(actual_max_core_indices_2.begin(), actual_max_core_indices_2.end());
    auto actual_max_clique_indices_2 =
        robin::FindInlierStructure(g2, robin::InlierGraphStructure::MAX_CLIQUE);
    std::sort(actual_max_clique_indices_2.begin(), actual_max_clique_indices_2.end());

    std::vector<size_t> expected_inliers_2 = {0, 1, 2, 4};
    REQUIRE_THAT(actual_max_core_indices, Catch::Equals(expected_inliers_2));
    REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers_2));

    delete g;
    delete g2;
  }

  SECTION("large normal noise bound equivalence") {
    //
    // This part tests the case where the error bound for normal is very large.
    // The resulting clique should be the same as a clique found just using points.
    //
    size_t N = 5;

    // random 3d points
    Eigen::Matrix<double, 3, Eigen::Dynamic> src_points =
        Eigen::Matrix<double, 3, Eigen::Dynamic>::Random(3, N);
    Eigen::Matrix<double, 4, Eigen::Dynamic> src_h;
    src_h.resize(4, src_points.cols());
    src_h.topRows(3) = src_points;
    src_h.bottomRows(1) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(N);

    // apply transformation
    Eigen::Matrix<double, 4, Eigen::Dynamic> tgt_h = T * src_h;
    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_points = tgt_h.topRows(3);

    // populate normals
    Eigen::Matrix<double, 3, Eigen::Dynamic> normals(3, N);
    normals = Eigen::Matrix<double, 3, Eigen::Dynamic>::Ones(3, N);

    // populate source
    Eigen::Matrix<double, 6, Eigen::Dynamic> src(6, N);
    src.topRows<3>() = src_points;
    src.bottomRows<3>() = normals;

    // add normal outlier (the corres. at index=3)
    size_t normal_outlier_idx = 3;
    normals(0, normal_outlier_idx) = 3;
    normals(1, normal_outlier_idx) = 5;
    normals(2, normal_outlier_idx) = 10;

    // add point outlier
    size_t point_outlier_idx = 4;
    tgt_points.col(point_outlier_idx) << 100, 100, 100;

    // populate target
    Eigen::Matrix<double, 6, Eigen::Dynamic> tgt(6, N);
    tgt.topRows<3>() = tgt_points;
    tgt.bottomRows<3>() = normals;

    double outlier_deviation_angle =
        std::acos((normals.col(normal_outlier_idx).dot(normals.col(0))) /
                  (normals.col(normal_outlier_idx).norm() * normals.col(0).norm()));

    // noise bound
    // with this noise bound, the point outlier should be identified.
    // however, the normal outlier should not be identified.
    Eigen::Vector2d noise_bound;
    noise_bound << 0.1, std::abs(outlier_deviation_angle) * 1.5;

    // expected inliers
    std::vector<size_t> expected_inliers = {0, 1, 2, 3};

    // call robin API
    auto* g = robin::Make3dNormalRegInvGraph(src, tgt, noise_bound);

    // find the clique
    auto actual_max_clique_indices =
        robin::FindInlierStructure(g, robin::InlierGraphStructure::MAX_CLIQUE);
    std::sort(actual_max_clique_indices.begin(), actual_max_clique_indices.end());
    REQUIRE_THAT(actual_max_clique_indices, Catch::Equals(expected_inliers));

    // check consistency with clique built from points only
    auto* g_points = robin::Make3dRegInvGraph(src_points, tgt_points, noise_bound(0));
    auto actual_points_max_clique_indices =
        robin::FindInlierStructure(g_points, robin::InlierGraphStructure::MAX_CLIQUE);
    std::sort(actual_points_max_clique_indices.begin(), actual_points_max_clique_indices.end());
    REQUIRE_THAT(actual_points_max_clique_indices, Catch::Equals(actual_max_clique_indices));

    delete g;
    delete g_points;
  }
}