// Copyright (c) 2020, Massachusetts Institute of Technology,
// Cambridge, MA 02139
// All Rights Reserved
// Authors: Jingnan Shi, et al. (see THANKS for the full author list)
// See LICENSE for the license information

#include "catch.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <iterator>
#include <set>
#include <iostream>

#include <robin/math.hpp>

TEST_CASE("n choose k") {

  SECTION("k=0") {
    int N = 500;
    for (int i = 0; i < N; ++i) {
      int result = robin::Choose(i, 0);
      REQUIRE(result == 1);
    }
  }

  SECTION("k=1") {
    int N = 500;
    for (int i = 1; i < N; ++i) {
      int result = robin::Choose(i, 1);
      REQUIRE(result == i);
    }
  }

  SECTION("k=n") {
    int N = 500;
    for (int i = 1; i < N; ++i) {
      int result = robin::Choose(i, i);
      REQUIRE(result == 1);
    }
  }

  SECTION("k>0") {
    REQUIRE(robin::Choose(2, 2) == 1);
    REQUIRE(robin::Choose(3, 2) == 3);
    REQUIRE(robin::Choose(10, 4) == 210);
    REQUIRE(robin::Choose(14, 3) == 364);
    REQUIRE(robin::Choose(14, 7) == 3432);
    REQUIRE(robin::Choose(42, 2) == 861);
    REQUIRE(robin::Choose(145, 5) == 498187404);
  }
}

TEST_CASE("Combination decode") {
  SECTION("2 choose 2") {
    size_t n = 2;
    size_t p = 2;
    std::set<std::vector<size_t>> exp_results;
    exp_results.insert({0,1});
    std::set<std::vector<size_t>> act_results;
    for (size_t i = 0; i < robin::Choose(n,p); ++i) {
      auto* result = new size_t [p];
      robin::CombinationDecode(n, p, i, result);
      std::vector<size_t> result_vec(result, result+p);
      act_results.insert(result_vec);
      bool found = exp_results.find(result_vec) != exp_results.end();
      REQUIRE(found);

      exp_results.erase(result_vec);
      delete [] result;
    }

    REQUIRE(exp_results.empty());
  }

  SECTION("3 choose 2") {
    size_t n = 3;
    size_t p = 2;
    std::set<std::vector<size_t>> exp_results;
    exp_results.insert({0,1});
    exp_results.insert({0,2});
    exp_results.insert({1,2});

    std::set<std::vector<size_t>> act_results;
    for (size_t i = 0; i < robin::Choose(n,p); ++i) {
      auto* result = new size_t [p];
      robin::CombinationDecode(n, p, i, result);
      std::vector<size_t> result_vec(result, result+p);
      act_results.insert(result_vec);
      bool found = exp_results.find(result_vec) != exp_results.end();
      REQUIRE(found);

      exp_results.erase(result_vec);
      delete [] result;
    }

    REQUIRE(exp_results.empty());
  }

  SECTION("4 choose 3") {
    size_t n = 4;
    size_t p = 3;
    std::set<std::vector<size_t>> exp_results;
    exp_results.insert({0,1,2});
    exp_results.insert({0,1,3});
    exp_results.insert({0,2,3});
    exp_results.insert({1,2,3});

    std::set<std::vector<size_t>> act_results;
    for (size_t i = 0; i < robin::Choose(n,p); ++i) {
      auto* result = new size_t [p];
      robin::CombinationDecode(n, p, i, result);
      std::vector<size_t> result_vec(result, result+p);
      act_results.insert(result_vec);
      bool found = exp_results.find(result_vec) != exp_results.end();
      REQUIRE(found);

      exp_results.erase(result_vec);
      delete [] result;
    }

    REQUIRE(exp_results.empty());
  }

  SECTION("5 choose 4") {
    size_t n = 5;
    size_t p = 4;
    std::set<std::vector<size_t>> exp_results;
    exp_results.insert({0,1,2,3});
    exp_results.insert({0,1,2,4});
    exp_results.insert({0,1,3,4});
    exp_results.insert({0,2,3,4});
    exp_results.insert({1,2,3,4});

    std::set<std::vector<size_t>> act_results;
    for (size_t i = 0; i < robin::Choose(n,p); ++i) {
      auto* result = new size_t [p];
      robin::CombinationDecode(n, p, i, result);
      std::vector<size_t> result_vec(result, result+p);
      act_results.insert(result_vec);
      bool found = exp_results.find(result_vec) != exp_results.end();
      REQUIRE(found);

      exp_results.erase(result_vec);
      delete [] result;
    }

    REQUIRE(exp_results.empty());
  }
}