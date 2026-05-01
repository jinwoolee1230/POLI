// Copyright (c) 2020, Massachusetts Institute of Technology,
// Cambridge, MA 02139
// All Rights Reserved
// Authors: Jingnan Shi, et al. (see THANKS for the full author list)
// See LICENSE for the license information

#include "catch.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <iterator>
#include <omp.h>
#include <set>

#include <robin/graph.hpp>
#include <robin/graph_io.hpp>
#include <robin/utils.hpp>

TEST_CASE("k-core solver methods") {
  // loading data
  robin::MatrixMarketReader reader;
  std::string test_data_folder = "./data/graph_test/";

  // graph 1: 5 vertices, 7 edges
  // 1-core: 4
  // 3-core: 0,1,2,3
  robin::AdjListGraph g = reader.readAdjListGraphFromFile(test_data_folder + "g1.mtx");
  const std::vector<size_t> g_expected_core = {3, 3, 3, 3, 1};
  const std::pair<robin::AdjListGraph, std::vector<size_t>> g_test_data(g, g_expected_core);

  // solve with BZ
  robin::KCoreDecompositionSolver kcore_solver(
      robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::BZ_SERIAL);
  kcore_solver.Solve(g);

  auto core_1 = kcore_solver.GetKCore(1);
  REQUIRE(core_1.size() == 5);

  auto core_3 = kcore_solver.GetKCore(3);
  REQUIRE(core_3.size() == 4);

  auto core_max = kcore_solver.GetMaxKCore();
  std::sort(core_max.begin(), core_max.end());
  std::sort(core_3.begin(), core_3.end());
  REQUIRE_THAT(core_max, Catch::Equals(core_3));

  auto core_max_number = kcore_solver.GetMaxCoreNumber();
  REQUIRE(core_max_number == 3);
}

TEST_CASE("k-core solver simple cases") {
  // loading data
  robin::MatrixMarketReader reader;
  std::string test_data_folder = "./data/graph_test/";

  // graph 1: 5 vertices, 7 edges
  // 1-core: 4
  // 3-core: 0,1,2,3
  robin::AdjListGraph g1_graph = reader.readAdjListGraphFromFile(test_data_folder + "g1.mtx");
  const std::vector<size_t> g1_expected_core = {3, 3, 3, 3, 1};
  const std::pair<robin::AdjListGraph, std::vector<size_t>> g1_test_data(g1_graph, g1_expected_core);

  // graph 2: 5 vertices, 0 edges
  robin::AdjListGraph g2_graph = reader.readAdjListGraphFromFile(test_data_folder + "g2.mtx");
  const std::vector<size_t> g2_expected_core = {0, 0, 0, 0, 0};
  const std::pair<robin::AdjListGraph, std::vector<size_t>> g2_test_data(g2_graph, g2_expected_core);

  // graph 2: 5 vertices, 4 edges
  // 1-core: 0,1,2,3,4
  robin::AdjListGraph g3_graph = reader.readAdjListGraphFromFile(test_data_folder + "g3.mtx");
  const std::vector<size_t> g3_expected_core = {1, 1, 1, 1, 1};
  const std::pair<robin::AdjListGraph, std::vector<size_t>> g3_test_data(g3_graph, g3_expected_core);

  auto single_runner = [&](auto mode, auto data_pair) {
    robin::KCoreDecompositionSolver kcore_solver(mode);
    kcore_solver.Solve(data_pair.first);
    for (size_t i = 0; i < kcore_solver.GetCoreNumbers().size(); ++i) {
      REQUIRE(data_pair.second[i] == kcore_solver.GetCoreNumbers()[i]);
    }
  };

  auto runner = [&](auto mode) {
    single_runner(mode, g1_test_data);
    single_runner(mode, g2_test_data);
    single_runner(mode, g3_test_data);
  };

  SECTION("BZ") { runner(robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::BZ_SERIAL); }

  SECTION("PKC Serial") { runner(robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_SERIAL); }

  SECTION("PKC Parallel") {
    runner(robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_PARALLEL);
  }

  SECTION("PKC Parallel Optimized") {
    runner(robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_PARALLEL_OPTIMIZED);
  }
}

TEST_CASE("k-core solver random graphs") {

  auto runner = [](size_t N) {
    auto g = robin::AdjListGraph::Random(N);
    robin::KCoreDecompositionSolver kcore_solver_bz(
        robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::BZ_SERIAL);
    robin::KCoreDecompositionSolver kcore_solver_pkc_serial(
        robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_SERIAL);
    robin::KCoreDecompositionSolver kcore_solver_pkc_parallel(
        robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_PARALLEL);
    robin::KCoreDecompositionSolver kcore_solver_pkc_parallel_optimized(
        robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_PARALLEL_OPTIMIZED);

    kcore_solver_bz.Solve(g);
    kcore_solver_pkc_serial.Solve(g);
    kcore_solver_pkc_parallel.Solve(g);
    kcore_solver_pkc_parallel_optimized.Solve(g);

    // true if solvers return core number lists of the same size
    REQUIRE(kcore_solver_bz.GetCoreNumbers().size() == N);
    REQUIRE(kcore_solver_pkc_serial.GetCoreNumbers().size() == N);
    REQUIRE(kcore_solver_pkc_parallel.GetCoreNumbers().size() == N);
    REQUIRE(kcore_solver_pkc_parallel_optimized.GetCoreNumbers().size() == N);

    // check whether the core numbers are the same
    for (size_t i = 0; i < N; ++i) {
      auto bz_core_num = kcore_solver_bz.GetCoreNumbers()[i];
      auto pkc_serial_core_num = kcore_solver_pkc_serial.GetCoreNumbers()[i];
      auto pkc_parallel_core_num = kcore_solver_pkc_parallel.GetCoreNumbers()[i];
      auto pkc_parallel_optim_core_num = kcore_solver_pkc_parallel_optimized.GetCoreNumbers()[i];

      std::vector<size_t> methods_cores_nums = {bz_core_num, pkc_serial_core_num,
                                                pkc_parallel_core_num, pkc_parallel_optim_core_num};
      bool all_equal = std::adjacent_find(methods_cores_nums.begin(), methods_cores_nums.end(),
                                          std::not_equal_to<>()) == methods_cores_nums.end();
      if (!all_equal) {
        std::cout << "Core numbers obtained: " << std::endl;
        std::cout << "BZ, PKC Ser, PKC Par, PKC Par Optim" << std::endl;
        std::cout << methods_cores_nums[0] << ", " << methods_cores_nums[1] << ", "
                  << methods_cores_nums[2] << ", " << methods_cores_nums[3] << std::endl;
      }
      REQUIRE(all_equal);
    }
  };

  SECTION("50 vertices") { runner(50); }
  SECTION("100 vertices") { runner(100); }
  SECTION("150 vertices") { runner(150); }
}

TEST_CASE("k-core solver large graphs") {
  robin::MatrixMarketReader reader;
  std::string test_data_folder = "./data/graph_test/";

  auto runner = [](const robin::AdjListGraph& g, size_t exp_max_k_core_size) {
    robin::KCoreDecompositionSolver kcore_solver_bz(
        robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::BZ_SERIAL);
    robin::KCoreDecompositionSolver kcore_solver_pkc_serial(
        robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_SERIAL);
    robin::KCoreDecompositionSolver kcore_solver_pkc_parallel(
        robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_PARALLEL);
    robin::KCoreDecompositionSolver kcore_solver_pkc_parallel_optimized(
        robin::KCoreDecompositionSolver::KCORE_SOLVER_MODE::PKC_PARALLEL_OPTIMIZED);

    kcore_solver_bz.Solve(g);
    kcore_solver_pkc_serial.Solve(g);
    kcore_solver_pkc_parallel.Solve(g);
    kcore_solver_pkc_parallel_optimized.Solve(g);

    // check whether the core numbers are the same
    for (size_t i = 0; i < g.VertexCount(); ++i) {
      auto bz_core_num = kcore_solver_bz.GetCoreNumbers()[i];
      auto pkc_serial_core_num = kcore_solver_pkc_serial.GetCoreNumbers()[i];
      auto pkc_parallel_core_num = kcore_solver_pkc_parallel.GetCoreNumbers()[i];
      auto pkc_parallel_optim_core_num = kcore_solver_pkc_parallel_optimized.GetCoreNumbers()[i];

      std::vector<size_t> methods_cores_nums = {bz_core_num, pkc_serial_core_num,
                                                pkc_parallel_core_num, pkc_parallel_optim_core_num};
      bool all_equal = std::adjacent_find(methods_cores_nums.begin(), methods_cores_nums.end(),
                                          std::not_equal_to<>()) == methods_cores_nums.end();
      if (!all_equal) {
        std::cout << "Core numbers obtained: " << std::endl;
        std::cout << "BZ, PKC Ser, PKC Par, PKC Par Optim" << std::endl;
        std::cout << methods_cores_nums[0] << ", " << methods_cores_nums[1] << ", "
                  << methods_cores_nums[2] << ", " << methods_cores_nums[3] << std::endl;
      }
      REQUIRE(all_equal);
    }

    // check max k core size
    REQUIRE(kcore_solver_bz.GetMaxCoreNumber() == exp_max_k_core_size);
    REQUIRE(kcore_solver_pkc_serial.GetMaxCoreNumber() == exp_max_k_core_size);
    REQUIRE(kcore_solver_pkc_parallel.GetMaxCoreNumber() == exp_max_k_core_size);
    REQUIRE(kcore_solver_pkc_parallel_optimized.GetMaxCoreNumber() == exp_max_k_core_size);
  };

  SECTION("web-edu") {
    robin::AdjListGraph g = reader.readAdjListGraphFromFile(test_data_folder + "web-edu.mtx");
    runner(g, 29);
  }
  SECTION("web-spam") {
    robin::AdjListGraph g = reader.readAdjListGraphFromFile(test_data_folder + "web-spam.mtx");
    runner(g, 35);
  }
  SECTION("c-fat500-10") {
    robin::AdjListGraph g = reader.readAdjListGraphFromFile(test_data_folder + "c-fat500-10.mtx");
    runner(g, 185);
  }
}

TEST_CASE("max clique small graphs") {
  SECTION("complete graph") {
    // A complete graph with max clique # = 5
    std::map<size_t, std::vector<size_t>> vertices_map;

    // create a complete graph
    int nodes_count = 5;
    for (int i = 0; i < nodes_count; ++i) {
      std::vector<size_t> temp;
      for (int j = 0; j < nodes_count; ++j) {
        if (j != i) {
          temp.push_back(j);
        }
      }
      vertices_map[i] = temp;
    }

    robin::AdjListGraph graph(vertices_map);
    robin::MaxCliqueSolver clique_solver;
    auto clique = clique_solver.FindMaxClique(graph);
    REQUIRE(clique.size() == 5);

    // Check whether the clique has the correct vertices
    std::cout << "Clique Nodes: ";
    std::copy(clique.begin(), clique.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    std::set<int> s(clique.begin(), clique.end());
    std::vector<int> ref_clique{0, 1, 2, 3, 4};
    for (const auto& i : ref_clique) {
      REQUIRE(s.find(i) != s.end());
    }
  }
  SECTION("isolated vertices") {
    // A graph with no edges
    robin::AdjListGraph graph;
    for (size_t i = 0; i < 10; ++i) {
      graph.AddVertex(i);
    }

    robin::MaxCliqueSolver clique_solver;
    auto clique = clique_solver.FindMaxClique(graph);
    REQUIRE(clique.empty());
  }
}

TEST_CASE("max clique large graphs single threaded") {
  robin::MatrixMarketReader reader;
  std::string test_data_folder = "./data/graph_test/";

  auto runner = [](const robin::AdjListGraph& g, size_t expected_lower_bound) {
    robin::MaxCliqueSolver clique_solver;
    auto clique = clique_solver.FindMaxClique(g);
    REQUIRE(clique.size() >= expected_lower_bound);
  };

  SECTION("web-edu") {
    omp_set_num_threads(1);
    robin::AdjListGraph g = reader.readAdjListGraphFromFile(test_data_folder + "web-edu.mtx");
    runner(g, 16);
  }
  SECTION("c-fat500-10") {
    omp_set_num_threads(1);
    robin::AdjListGraph g = reader.readAdjListGraphFromFile(test_data_folder + "c-fat500-10.mtx");
    runner(g, 126);
  }
}

TEST_CASE("max clique multiple threads") {
  robin::MatrixMarketReader reader;
  std::string test_data_folder = "./data/graph_test/";

  auto runner = [](const robin::AdjListGraph& g, size_t expected_lower_bound) {
    robin::MaxCliqueSolver clique_solver;
    auto clique = clique_solver.FindMaxClique(g);
    REQUIRE(clique.size() >= expected_lower_bound);
  };

  SECTION("web-edu") {
    omp_set_num_threads(2);
    robin::AdjListGraph g = reader.readAdjListGraphFromFile(test_data_folder + "web-edu.mtx");
    runner(g, 16);
  }
  SECTION("c-fat500-10") {
    omp_set_num_threads(2);
    robin::AdjListGraph g = reader.readAdjListGraphFromFile(test_data_folder + "c-fat500-10.mtx");
    runner(g, 126);
  }
}
